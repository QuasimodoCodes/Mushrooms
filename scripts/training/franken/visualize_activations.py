"""
visualize_activations.py — pass one image through TaxonomicYOLO26 and save
a PNG showing what each backbone layer "sees".

Usage
-----
    python visualize_activations.py path/to/image.jpg
    python visualize_activations.py path/to/image.jpg --out docs/franken_runs/activations/
    python visualize_activations.py path/to/image.jpg --weights docs/franken_runs/taxonomic_yolo26_5/weights/best.pt
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _HERE)

from model import TaxonomicYOLO26  # noqa: E402
from predict import SPECIES, GENERA  # noqa: E402

# ---------------------------------------------------------------------------
# Same eval transform as training
# ---------------------------------------------------------------------------
_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

_UNNORM_MEAN = np.array([0.485, 0.456, 0.406])
_UNNORM_STD  = np.array([0.229, 0.224, 0.225])


def unnormalize(tensor):
    """Convert normalised tensor back to a displayable numpy image."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * _UNNORM_STD + _UNNORM_MEAN
    return np.clip(img, 0, 1)


def activation_to_grid(act, n_cols=8, max_channels=32):
    """
    Given a 4-D activation tensor (1, C, H, W) return a tiled numpy image
    showing up to max_channels individual channel maps.
    """
    act = act[0].cpu().float()          # (C, H, W)
    C = min(act.shape[0], max_channels)
    act = act[:C]

    # Normalise each channel to [0, 1] independently
    mins = act.flatten(1).min(1).values[:, None, None]
    maxs = act.flatten(1).max(1).values[:, None, None]
    act  = (act - mins) / (maxs - mins + 1e-8)

    n_rows = (C + n_cols - 1) // n_cols
    H, W   = act.shape[1], act.shape[2]
    canvas = np.zeros((n_rows * H, n_cols * W), dtype=np.float32)

    for i in range(C):
        r, c = divmod(i, n_cols)
        canvas[r*H:(r+1)*H, c*W:(c+1)*W] = act[i].numpy()

    return canvas


def run(image_path, weights_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------ model
    print("Loading model...")
    model = TaxonomicYOLO26(
        num_genera=len(GENERA),
        num_species=len(SPECIES),
        weights=os.path.join(_ROOT, "yolo26n-cls.pt"),
    )
    ckpt  = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.to(device).eval()

    # ------------------------------------------------------------------ image
    img_pil = Image.open(image_path).convert("RGB")
    tensor  = _TRANSFORM(img_pil).unsqueeze(0).to(device)

    # --------------------------------------------------------- capture layers
    # Re-run _extract_features but capture activations after each backbone layer.
    backbone_layers = list(model._inner.model)[:-1]

    captured = []   # list of (layer_name, spatial_tensor_or_None)

    x = tensor
    y = []
    with torch.no_grad():
        for m in backbone_layers:
            if m.f != -1:
                x = (y[m.f] if isinstance(m.f, int)
                     else [x if j == -1 else y[j] for j in m.f])
            x = m(x)
            y.append(x if m.i in model._inner.save else None)

            # Only keep 4-D tensors (B, C, H, W) for visualisation
            out = x
            if isinstance(out, (list, tuple)):
                out = out[0]
            if isinstance(out, torch.Tensor) and out.dim() == 4:
                captured.append((type(m).__name__, out.detach()))

        # Feature vector (after flatten)
        feat = x
        if isinstance(feat, (list, tuple)):
            feat = torch.cat([t.flatten(1) for t in feat], dim=1)
        else:
            feat = feat.flatten(1)

        # Heads (eval mode — dropout off)
        genus_logits, species_logits = model.genus_head(feat), model.species_head(feat)

    genus_probs   = F.softmax(genus_logits,   dim=1)[0].cpu()
    species_probs = F.softmax(species_logits, dim=1)[0].cpu()

    print(f"Captured {len(captured)} spatial activation layers")

    # ------------------------------------------------------- decide which layers to plot
    # Pick ~6 evenly spaced layers from captured
    n_plot = min(6, len(captured))
    indices = np.linspace(0, len(captured) - 1, n_plot, dtype=int).tolist()
    plot_layers = [captured[i] for i in indices]

    # ----------------------------------------------------------------- figure
    # Layout:
    #   Row 0:  original image  |  6 activation grids
    #   Row 1:  feature vector heatmap (full width)
    #   Row 2:  genus top-10 bar   |   species top-10 bar

    fig = plt.figure(figsize=(22, 14), facecolor="#0f0f0f")
    fig.suptitle(
        f"TaxonomicYOLO26 — Layer Activations\n{os.path.basename(image_path)}",
        color="white", fontsize=14, y=0.98,
    )

    outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[4, 1.5, 3.5],
                              hspace=0.35)

    # ---- Row 0: image + activation grids ----
    top = gridspec.GridSpecFromSubplotSpec(
        1, n_plot + 1, subplot_spec=outer[0], wspace=0.04
    )

    # Original image
    ax_img = fig.add_subplot(top[0])
    ax_img.imshow(np.array(img_pil.resize((224, 224))))
    ax_img.set_title("Input\n224×224", color="white", fontsize=8)
    ax_img.axis("off")

    # Activation grids
    for col, (layer_name, act) in enumerate(plot_layers, start=1):
        grid = activation_to_grid(act, n_cols=8, max_channels=32)
        ax = fig.add_subplot(top[col])
        ax.imshow(grid, cmap="inferno", aspect="auto")
        C, H, W = act.shape[1], act.shape[2], act.shape[3]
        depth_label = ["Early\n(edges)", "Early-mid\n(textures)",
                       "Mid\n(patterns)", "Mid-late\n(shapes)",
                       "Late\n(semantics)", "Pre-pool\n(abstract)"]
        label = depth_label[col - 1] if col - 1 < len(depth_label) else layer_name
        ax.set_title(f"{label}\n{C}ch {H}×{W}", color="white", fontsize=7)
        ax.axis("off")

    # ---- Row 1: feature vector heatmap ----
    ax_feat = fig.add_subplot(outer[1])
    feat_np = feat[0].cpu().float().numpy()
    feat_np = (feat_np - feat_np.min()) / (feat_np.max() - feat_np.min() + 1e-8)
    ax_feat.imshow(feat_np[np.newaxis, :], cmap="viridis",
                   aspect="auto", vmin=0, vmax=1)
    ax_feat.set_title(
        f"Feature Vector ({feat.shape[1]} dims) — the backbone's summary of this image",
        color="white", fontsize=9,
    )
    ax_feat.set_yticks([])
    ax_feat.tick_params(colors="white", labelsize=7)
    for spine in ax_feat.spines.values():
        spine.set_edgecolor("#444")

    # ---- Row 2: genus + species bar charts ----
    bottom = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[2], wspace=0.3
    )

    topk = 10

    # Genus
    ax_g = fig.add_subplot(bottom[0])
    g_topk = genus_probs.topk(topk)
    g_labels = [GENERA[i] for i in g_topk.indices]
    g_vals   = g_topk.values.numpy() * 100
    colors_g = ["#f0a500" if i == 0 else "#555" for i in range(topk)]
    bars = ax_g.barh(range(topk - 1, -1, -1), g_vals, color=colors_g)
    ax_g.set_yticks(range(topk - 1, -1, -1))
    ax_g.set_yticklabels(g_labels, color="white", fontsize=8)
    ax_g.set_xlabel("Confidence (%)", color="white", fontsize=8)
    ax_g.set_title("Genus Head — Top 10", color="white", fontsize=10)
    ax_g.tick_params(colors="white")
    ax_g.set_facecolor("#1a1a1a")
    for spine in ax_g.spines.values():
        spine.set_edgecolor("#444")
    # Label the top bar
    ax_g.text(g_vals[0] + 0.5, topk - 1, f"{g_vals[0]:.1f}%",
              va="center", color="#f0a500", fontsize=8, fontweight="bold")

    # Species
    ax_s = fig.add_subplot(bottom[1])
    s_topk = species_probs.topk(topk)
    s_labels = [SPECIES[i] for i in s_topk.indices]
    s_vals   = s_topk.values.numpy() * 100
    colors_s = ["#00c49a" if i == 0 else "#555" for i in range(topk)]
    ax_s.barh(range(topk - 1, -1, -1), s_vals, color=colors_s)
    ax_s.set_yticks(range(topk - 1, -1, -1))
    ax_s.set_yticklabels(
        [f"{'*' if l.split()[0] == GENERA[g_topk.indices[0]] else ' '} {l}"
         for l in s_labels],
        color="white", fontsize=8,
    )
    ax_s.set_xlabel("Confidence (%)", color="white", fontsize=8)
    ax_s.set_title("Species Head — Top 10   (* = matches genus pick)", color="white", fontsize=10)
    ax_s.tick_params(colors="white")
    ax_s.set_facecolor("#1a1a1a")
    for spine in ax_s.spines.values():
        spine.set_edgecolor("#444")
    ax_s.text(s_vals[0] + 0.5, topk - 1, f"{s_vals[0]:.1f}%",
              va="center", color="#00c49a", fontsize=8, fontweight="bold")

    fig.patch.set_facecolor("#0f0f0f")

    # ------------------------------------------------------------------ save
    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_activations.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved: {out_path}")

    # Console summary
    print(f"\nGenus:    {GENERA[g_topk.indices[0]]}  ({g_vals[0]:.1f}%)")
    print(f"Species:  {SPECIES[s_topk.indices[0]]}  ({s_vals[0]:.1f}%)")
    top_sp_genus = SPECIES[s_topk.indices[0]].split()[0]
    top_genus    = GENERA[g_topk.indices[0]]
    if top_genus == top_sp_genus:
        print("[OK] Heads agree")
    else:
        print(f"[!] Heads disagree — species implies '{top_sp_genus}', genus says '{top_genus}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--weights",
        default=os.path.join(_ROOT, "docs", "franken_runs",
                             "taxonomic_yolo26_5", "weights", "best.pt"),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(_ROOT, "docs", "franken_runs", "activations"),
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"ERROR: image not found: {args.image}")
        sys.exit(1)

    run(args.image, args.weights, args.out)


if __name__ == "__main__":
    main()
