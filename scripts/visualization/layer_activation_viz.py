"""
Layer Activation Visualisation
================================
Shows what each stage of a model "sees" as the image passes through it.
At each major layer, captures the feature maps and displays:
  - Mean activation (spatial heatmap — where is the model looking?)
  - Top-8 individual channel activations (what patterns is it detecting?)
  - A plain-English explanation of what that stage does

One model at a time. One PNG output per run.

Usage (from project root):
    python scripts/visualization/layer_activation_viz.py --model yolo      --image path/to/img.jpg
    python scripts/visualization/layer_activation_viz.py --model efficientnet --image path/to/img.jpg
    python scripts/visualization/layer_activation_viz.py --model taxonomic  --image path/to/img.jpg

Output:
    docs/layer_viz_<model>.png
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from PIL import Image
from torchvision import transforms

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.abspath(os.path.join(_HERE, "..", ".."))

YOLO_WEIGHTS    = os.path.join(_ROOT, "docs", "yolo_runs",    "yolo26_classifier_v1",  "weights", "best.pt")
YOLO8_WEIGHTS   = os.path.join(_ROOT, "docs", "yolo_runs",    "mushroom_classifier_v1", "weights", "best.pt")
EFFNET_WEIGHTS  = os.path.join(_ROOT, "docs", "cnn_runs",     "efficientnet_b0_adamw_ce_smooth", "weights", "best.pt")
FRANKEN_WEIGHTS = os.path.join(_ROOT, "docs", "franken_runs", "taxonomic_yolo26_5",    "weights", "best.pt")
CLASS_NAMES_JSON= os.path.join(_ROOT, "docs", "yolo_runs",    "yolo26_tflite",         "weights", "class_names.json")

sys.path.insert(0, os.path.join(_HERE, "..", "training", "franken"))

# ── Preprocessing ──────────────────────────────────────────────────────────────
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def load_image(path: str, device: str):
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    pil = Image.open(path).convert("RGB")
    pil = transforms.Resize(256)(pil)
    pil = transforms.CenterCrop(IMG_SIZE)(pil)
    return tfm(Image.open(path).convert("RGB")).to(device), pil


def pil_to_np(pil_img) -> np.ndarray:
    return np.array(pil_img).astype(np.float32) / 255.0


def norm(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


# ── Feature-map helpers ────────────────────────────────────────────────────────

def mean_activation(fmap: torch.Tensor) -> np.ndarray:
    """Average all channels → one spatial heatmap (H×W)."""
    act = fmap.squeeze(0).abs().mean(dim=0).cpu().numpy()
    return norm(act)


def top_channels(fmap: torch.Tensor, k: int = 8):
    """Return the k most strongly activated channel maps, each normalised."""
    maps = fmap.squeeze(0).cpu()                        # (C, H, W)
    energy = maps.abs().mean(dim=[1, 2])                # (C,)
    indices = energy.topk(min(k, maps.shape[0])).indices
    return [norm(maps[i].numpy()) for i in indices]


# ── Stage annotations ──────────────────────────────────────────────────────────
# Each entry: (hook_path, short_title, plain_english_explanation)

YOLO_STAGES = [
    ("model.0",
     "Stem — Conv 3×3\n32 filters, stride 2",
     "Very first layer. Learns simple edges, colour gradients, and\n"
     "basic textures — the same low-level features every CNN learns first.\n"
     "Output is half the input size (112×112)."),

    ("model.2",
     "C3k2 Block 1\n64 filters",
     "Cross-stage partial block. Splits channels, processes each branch\n"
     "separately, then merges. Starts detecting corners, blobs, and\n"
     "surface texture — cap rim, background clutter."),

    ("model.4",
     "C3k2 Block 2\n128 filters",
     "Deeper pattern detection. Feature maps shrink (56×56) but each\n"
     "channel encodes a richer concept — colour patches, gill lines,\n"
     "stem base shapes. Background starts to fade."),

    ("model.6",
     "C3k2 Block 3\n256 filters",
     "High-level spatial reasoning (28×28). The model has now built\n"
     "representations for whole cap shapes, stem-cap junctions, and\n"
     "environmental context like leaf litter."),

    ("model.8",
     "C3k2 Block 4\n512 filters",
     "Near-semantic features (14×14). Individual channels here correspond\n"
     "to meaningful visual concepts — 'shiny cap', 'white gills visible',\n"
     "'bulbous base'. This is where species discrimination happens."),

    ("model.9",
     "C2PSA — Attention\nMulti-head self-attention",
     "Position-sensitive attention block unique to YOLOv26. Unlike\n"
     "standard YOLO, this layer lets spatial positions attend to each\n"
     "other — similar to a Transformer. Captures long-range dependencies\n"
     "like gill-to-cap relationships across the image."),
]

EFFNET_STAGES = [
    ("features.0",
     "Stem — Conv 3×3\n32 filters, SiLU activation",
     "First convolution. Detects basic edges and colour transitions.\n"
     "SiLU (Swish) activation is smoother than ReLU — preserves\n"
     "small gradient signals that ReLU would discard."),

    ("features.2",
     "MBConv Block 2\nExpand → Depthwise → Squeeze-Excite",
     "Mobile Inverted Bottleneck. Expands channels, applies a depthwise\n"
     "convolution (one filter per channel — very efficient), then\n"
     "Squeeze-Excitation re-weights channels by global importance.\n"
     "Starts separating foreground mushroom from background."),

    ("features.4",
     "MBConv Block 4\n40 filters, 28×28",
     "Mid-level shape detection. Cap curvature, stem roundness, and\n"
     "gill patterns begin emerging as distinct channel responses.\n"
     "SE block emphasises species-relevant channels."),

    ("features.6",
     "MBConv Block 6\n192 filters, 14×14",
     "Rich feature representation. Channels now encode high-level\n"
     "visual concepts — wart colour, gill attachment, ring presence.\n"
     "This is where EfficientNet and YOLO diverge most: compound\n"
     "scaling gives EfficientNet wider channels at this depth."),

    ("features.8",
     "Final MBConv Block\n1280 filters, 7×7",
     "Final spatial feature map before the classifier head.\n"
     "1280 channels distilled to a 7×7 grid. Each of the 49 spatial\n"
     "positions holds a 1280-dimensional description of that region.\n"
     "This is what GradCAM reads for its heatmap."),
]

FRANKEN_STAGES = [
    ("_inner.model.0",
     "Shared Stem\nSame backbone as YOLOv26",
     "Identical first layer to the standalone YOLO model — the backbone\n"
     "is shared between both heads. What you see here is exactly what\n"
     "YOLO's first layer also sees. No difference yet."),

    ("_inner.model.4",
     "Shared C3k2 Block 2\nBoth heads read this",
     "Mid-level features used by BOTH genus and species heads.\n"
     "The backbone must learn features useful for coarse genus\n"
     "classification AND fine species classification simultaneously.\n"
     "This dual pressure is the regularisation effect."),

    ("_inner.model.8",
     "Shared Deep Features\nHigh-level representations",
     "Near the top of the shared backbone. Both heads pull from\n"
     "this representation. Because genus classification is easier,\n"
     "its gradient signal stabilises training here and prevents\n"
     "the species head from over-specialising."),
]

YOLO8_STAGES = [
    ("model.0",
     "Stem — Conv 3×3\n32 filters, stride 2",
     "First convolution — identical role to YOLOv26's stem.\n"
     "Learns edges, colour gradients, basic textures.\n"
     "Output: 112×112. Baseline to compare against v26."),

    ("model.2",
     "C2f Block 1\n64 filters",
     "YOLOv8's cross-stage partial block. Uses fewer bottleneck\n"
     "layers than YOLOv26's C3k2 — lighter but slightly less\n"
     "expressive. Detects corners, blobs, surface texture."),

    ("model.4",
     "C2f Block 2\n128 filters",
     "Mid-level shape features (56×56). Cap curvature, stem\n"
     "outlines and gill patterns begin to emerge. Compare with\n"
     "YOLOv26 Block 2 — activations are structurally similar\n"
     "but YOLOv8 uses 2.7M parameters vs 1.74M for v26."),

    ("model.6",
     "C2f Block 3\n256 filters",
     "High-level spatial reasoning (28×28). Whole cap shapes,\n"
     "stem-cap junctions, and environmental context build up\n"
     "here. YOLOv8 reaches this abstraction one layer earlier\n"
     "than YOLOv26 due to its simpler block design."),

    ("model.8",
     "C2f Block 4\n512 filters",
     "Deepest spatial feature map before the classifier (14×14).\n"
     "Species-discriminative features live here — 'shiny cap',\n"
     "'visible gills', 'bulbous base'. This is the layer where\n"
     "YOLOv8 and YOLOv26 diverge most in learned representations."),
]


# ── Hook collector ────────────────────────────────────────────────────────────

class ActivationCollector:
    def __init__(self):
        self._captures = {}
        self._hooks    = []

    def register(self, layer: torch.nn.Module, name: str):
        def hook(_m, _in, out):
            act = out[0] if isinstance(out, tuple) else out
            if act.dim() >= 3:   # only keep spatial feature maps
                self._captures[name] = act.detach()
        self._hooks.append(layer.register_forward_hook(hook))

    def captures(self):
        return self._captures

    def remove(self):
        for h in self._hooks:
            h.remove()


def _get_nested(module, path: str):
    """Navigate dotted path: 'features.2' → module.features[2]."""
    parts = path.split(".")
    obj = module
    for p in parts:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return obj


# ── Per-model runners ──────────────────────────────────────────────────────────

def run_yolo(tensor, class_names, device):
    from ultralytics import YOLO
    yolo  = YOLO(YOLO_WEIGHTS)
    model = yolo.model.to(device).eval()

    collector = ActivationCollector()
    for path, title, desc in YOLO_STAGES:
        try:
            layer = _get_nested(model, path)
            collector.register(layer, path)
        except Exception:
            pass

    with torch.no_grad():
        logits = model(tensor.unsqueeze(0))
    if isinstance(logits, (list, tuple)):
        logits = logits[-1]
    pred_idx   = logits.argmax(1).item()
    confidence = F.softmax(logits, dim=1)[0, pred_idx].item()
    label      = class_names.get(pred_idx, f"class_{pred_idx}")

    collector.remove()
    return collector.captures(), YOLO_STAGES, label, confidence


def run_yolo8(tensor, class_names, device):
    from ultralytics import YOLO
    yolo  = YOLO(YOLO8_WEIGHTS)
    model = yolo.model.to(device).eval()

    collector = ActivationCollector()
    for path, title, desc in YOLO8_STAGES:
        try:
            layer = _get_nested(model, path)
            collector.register(layer, path)
        except Exception:
            pass

    with torch.no_grad():
        logits = model(tensor.unsqueeze(0))
    if isinstance(logits, (list, tuple)):
        logits = logits[-1]
    pred_idx   = logits.argmax(1).item()
    confidence = F.softmax(logits, dim=1)[0, pred_idx].item()
    label      = class_names.get(pred_idx, f"class_{pred_idx}")

    collector.remove()
    return collector.captures(), YOLO8_STAGES, label, confidence


def run_effnet(tensor, class_names, device):
    import torch.nn as nn
    from torchvision import models

    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(m.classifier[1].in_features, len(class_names)),
    )
    m.load_state_dict(torch.load(EFFNET_WEIGHTS, map_location=device))
    m = m.to(device).eval()

    collector = ActivationCollector()
    for path, title, desc in EFFNET_STAGES:
        try:
            layer = _get_nested(m, path)
            collector.register(layer, path)
        except Exception:
            pass

    with torch.no_grad():
        logits = m(tensor.unsqueeze(0))
    pred_idx   = logits.argmax(1).item()
    confidence = F.softmax(logits, dim=1)[0, pred_idx].item()
    label      = class_names.get(pred_idx, f"class_{pred_idx}")

    collector.remove()
    return collector.captures(), EFFNET_STAGES, label, confidence


def run_taxonomic(tensor, class_names, device):
    from model import TaxonomicYOLO26

    ckpt  = torch.load(FRANKEN_WEIGHTS, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    num_genera  = state["genus_head.1.weight"].shape[0]

    model = TaxonomicYOLO26(
        num_genera=num_genera,
        num_species=len(class_names),
        weights=YOLO_WEIGHTS,
    ).to(device).eval()

    model_state = model.state_dict()
    filtered    = {k: v for k, v in state.items()
                   if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)

    collector = ActivationCollector()
    for path, title, desc in FRANKEN_STAGES:
        try:
            layer = _get_nested(model, path)
            collector.register(layer, path)
        except Exception:
            pass

    with torch.no_grad():
        genus_logits, species_logits = model(tensor.unsqueeze(0))

    s_pred  = species_logits.argmax(1).item()
    s_conf  = F.softmax(species_logits, dim=1)[0, s_pred].item()
    g_pred  = genus_logits.argmax(1).item()
    g_conf  = F.softmax(genus_logits,  dim=1)[0, g_pred].item()
    label   = f"Species: {class_names.get(s_pred, f'class_{s_pred}')} ({s_conf*100:.1f}%)\nGenus head: genus_{g_pred} ({g_conf*100:.1f}%)"

    collector.remove()
    return collector.captures(), FRANKEN_STAGES, label, 0.0


# ── Figure builder ─────────────────────────────────────────────────────────────

BG   = "#f5ede0"   # warm cream (slide background)
CARD = "#ffffff"   # white cards
ACC  = "#2d5a27"   # dark forest green (header / icons)
GOLD = "#c8a040"   # amber gold (left accent bar on slide)
TXT  = "#1c1c1c"   # near-black for title
MUTED= "#4a5540"   # olive-toned gray for body text
HEAT = "magma"     # activation heatmap — stands out on light bg

def build_figure(model_name: str, img_np: np.ndarray, captures: dict,
                 stages: list, label: str, confidence: float, out_path: str):

    import textwrap

    valid_stages = [(p, t, d) for p, t, d in stages if p in captures]
    n_stages     = len(valid_stages)
    CHANNELS     = 4   # top-N channels per stage

    total_cols = 1 + n_stages
    # ~3 inches per stage column at 200 dpi keeps pixels manageable but fonts large
    fig_w = 4.0 + n_stages * 3.2
    fig_h = 14
    fig   = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)
    DPI   = 200

    title_map = {
        "yolo":         "YOLOv26n-cls — Layer Activation Visualisation",
        "yolo8":        "YOLOv8n-cls — Layer Activation Visualisation",
        "efficientnet": "EfficientNet-B0 — Layer Activation Visualisation",
        "taxonomic":    "TaxonomicYOLO26 — Layer Activation Visualisation\n(shared backbone — genus + species heads both read these features)",
    }
    # Count suptitle lines to adjust spacing automatically
    n_title_lines = title_map.get(model_name, model_name).count("\n") + 1
    top_margin = 0.950 if n_title_lines == 1 else 0.930   # more room for 2-line title

    fig.suptitle(title_map.get(model_name, model_name),
                 color=TXT, fontsize=14, fontweight="bold", y=0.997)

    # ── Outer layout: feature maps (tall) + description cards (compact) ──────
    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[3.5, 1.0],
        top=top_margin, bottom=0.01,
        hspace=0.08,
    )

    # ── Top section: input image + stage columns ──────────────────────────────
    top_grid = gridspec.GridSpecFromSubplotSpec(
        1, total_cols,
        subplot_spec=outer[0],
        wspace=0.05,
    )

    # Input image
    ax_orig = fig.add_subplot(top_grid[0])
    ax_orig.imshow(img_np)
    pred_text = label if confidence == 0.0 else f"{label}\n{confidence*100:.1f}%"
    ax_orig.set_title(f"Input Image\n\nPrediction:\n{pred_text}",
                      color=ACC, fontsize=10, pad=6, loc="left")
    ax_orig.axis("off")

    # Stage feature-map columns
    for col_idx, (path, title, desc) in enumerate(valid_stages):
        fmap = captures[path]

        inner = gridspec.GridSpecFromSubplotSpec(
            3, CHANNELS,
            subplot_spec=top_grid[col_idx + 1],
            hspace=0.04, wspace=0.03,
        )

        # Row 0: mean activation heatmap (full width of column)
        ax_mean = fig.add_subplot(inner[0, :])
        ax_mean.imshow(mean_activation(fmap), cmap=HEAT, aspect="auto")
        c_str = f"{fmap.shape[1]} ch · {fmap.shape[2]}×{fmap.shape[3]} px"
        ax_mean.set_title(f"{title}\n{c_str}", color=ACC,
                          fontsize=9, fontweight="bold", pad=4)
        ax_mean.axis("off")

        # Rows 1–2: individual top channels
        channels = top_channels(fmap, k=CHANNELS * 2)
        for ci, ch_map in enumerate(channels[:CHANNELS * 2]):
            r = 1 + ci // CHANNELS
            c = ci  % CHANNELS
            ax = fig.add_subplot(inner[r, c])
            ax.imshow(ch_map, cmap=HEAT, aspect="auto")
            ax.axis("off")

    # ── Bottom section: description cards ────────────────────────────────────
    # Card height in figure-fraction units for font sizing reference
    card_h_in = fig_h * (1.0 / (3.5 + 1.0))  # bottom ratio share of fig_h

    # Wrap width proportional to column width (chars that fit comfortably)
    wrap_w = max(18, int(3.2 * 8))   # ~8 chars per inch at this font size

    bot_grid = gridspec.GridSpecFromSubplotSpec(
        1, n_stages,
        subplot_spec=outer[1],
        wspace=0.05,
    )

    for i, (path, title, desc) in enumerate(valid_stages):
        ax_d = fig.add_subplot(bot_grid[i])
        ax_d.set_facecolor(CARD)
        ax_d.set_xlim(0, 1)
        ax_d.set_ylim(0, 1)
        ax_d.axis("off")
        # Subtle card border
        for spine in ["top", "bottom", "left", "right"]:
            ax_d.spines[spine].set_visible(True)
            ax_d.spines[spine].set_color("#d8cfc4")
            ax_d.spines[spine].set_linewidth(0.8)

        # Gold accent bar at top (matches slide left-sidebar colour)
        ax_d.axhline(y=0.97, color=GOLD, linewidth=3.0)

        # Stage title (bold, forest-green) — starts just below accent line
        ax_d.text(0.04, 0.93, title,
                  transform=ax_d.transAxes,
                  color=ACC, fontsize=10, fontweight="bold",
                  va="top", linespacing=1.3)

        # Description text — wrap each paragraph to fit the column
        wrapped = "\n".join(
            textwrap.fill(para, width=wrap_w)
            for para in desc.split("\n")
            if para.strip()
        )
        ax_d.text(0.04, 0.70, wrapped,
                  transform=ax_d.transAxes,
                  color=MUTED, fontsize=9.5,
                  va="top", linespacing=1.5)

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  >> Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True,
                        choices=["yolo", "yolo8", "efficientnet", "taxonomic"],
                        help="Which model to visualise")
    parser.add_argument("--image",  required=True, help="Path to input image")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model  : {args.model}")
    print(f"Device : {device}")
    print(f"Image  : {args.image}")

    with open(CLASS_NAMES_JSON) as f:
        class_names = {int(k): v for k, v in json.load(f).items()}

    tensor, pil_img = load_image(args.image, device)
    img_np = pil_to_np(pil_img)

    out_path = os.path.join(_ROOT, "docs", f"layer_viz_{args.model}.png")

    runners = {
        "yolo":        run_yolo,
        "yolo8":       run_yolo8,
        "efficientnet":run_effnet,
        "taxonomic":   run_taxonomic,
    }

    captures, stages, label, conf = runners[args.model](tensor, class_names, device)
    print(f"Prediction: {label}  {conf*100:.1f}%" if conf > 0 else f"Prediction: {label}")
    print(f"Captured {len(captures)} / {len(stages)} stages")

    build_figure(args.model, img_np, captures, stages, label, conf, out_path)
    print(f"\nDone. Open: {out_path}")


if __name__ == "__main__":
    main()
