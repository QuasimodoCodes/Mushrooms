"""
Model Comparison — GradCAM Visualisation
==========================================
Runs the same image through all three trained models and generates a side-by-side
GradCAM heatmap showing what each model focuses on to make its decision.

Output layout (one row per model):
    Original | Grad-CAM Heatmap | Overlay + Prediction

Models:
    1. YOLOv26n-cls       — production backbone
    2. EfficientNet-B0    — ImageNet-pretrained classification-native architecture
    3. TaxonomicYOLO26    — dual-head (genus + species), two heatmaps shown

Usage (from project root):
    python scripts/visualization/model_comparison_gradcam.py --image path/to/mushroom.jpg

Output:
    docs/model_comparison_gradcam.png
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
from PIL import Image
from torchvision import transforms

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.abspath(os.path.join(_HERE, "..", ".."))

YOLO_WEIGHTS    = os.path.join(_ROOT, "docs", "yolo_runs",   "yolo26_classifier_v1", "weights", "best.pt")
EFFNET_WEIGHTS  = os.path.join(_ROOT, "docs", "cnn_runs",    "efficientnet_b0_adamw_ce_smooth", "weights", "best.pt")
FRANKEN_WEIGHTS = os.path.join(_ROOT, "docs", "franken_runs","taxonomic_yolo26_5",    "weights", "best.pt")
CLASS_NAMES_JSON= os.path.join(_ROOT, "docs", "yolo_runs",   "yolo26_tflite", "weights", "class_names.json")
FRANKEN_GENERA  = os.path.join(_ROOT, "docs", "franken_runs","taxonomic_yolo26_5", "genera.json")
OUT_PATH        = os.path.join(_ROOT, "docs", "model_comparison_gradcam.png")

# Add franken dir to path (for TaxonomicYOLO26 model definition)
sys.path.insert(0, os.path.join(_HERE, "..", "training", "franken"))


# ── Image preprocessing ────────────────────────────────────────────────────────
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def load_image(path: str, device: str):
    """Returns (preprocessed tensor, PIL image cropped to 224×224)."""
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    pil = Image.open(path).convert("RGB")
    pil = transforms.Resize(256)(pil)
    pil = transforms.CenterCrop(IMG_SIZE)(pil)
    tensor = tfm(Image.open(path).convert("RGB")).to(device)
    return tensor, pil


def pil_to_np(pil_img) -> np.ndarray:
    return np.array(pil_img).astype(np.float32) / 255.0


def overlay(img_np: np.ndarray, cam: np.ndarray, alpha=0.45) -> np.ndarray:
    heatmap = plt.cm.jet(cam)[:, :, :3]
    return np.clip(img_np * (1 - alpha) + heatmap * alpha, 0, 1)


# ── Generic GradCAM ────────────────────────────────────────────────────────────

class GradCAM:
    """
    Attaches forward + backward hooks to a target layer.
    Works with any nn.Module layer that outputs a (B, C, H, W) feature map.
    """
    def __init__(self, target_layer: torch.nn.Module):
        self._acts = None
        self._grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _in, out):
        # Handle tuple outputs (some YOLO layers return tuples)
        self._acts = (out[0] if isinstance(out, tuple) else out).detach()

    def _bwd(self, _m, _in, grad_out):
        self._grads = (grad_out[0] if isinstance(grad_out, tuple) else grad_out[0]).detach()

    def compute(self) -> np.ndarray:
        """Returns normalised CAM, shape (224, 224)."""
        if self._acts is None or self._grads is None:
            raise RuntimeError("No activations/gradients captured — did forward+backward run?")
        # If spatial dims collapsed (1D), we can't do spatial GradCAM
        if self._acts.dim() < 4:
            raise RuntimeError("Target layer output has no spatial dims — pick an earlier layer.")
        weights = self._grads.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self._acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ── Model 1: YOLOv26n-cls ─────────────────────────────────────────────────────

def run_yolo_gradcam(tensor: torch.Tensor, class_names: list, device: str):
    """
    Hook into the second-to-last YOLO layer (before the Classify head).
    Returns (cam, predicted_label, confidence).
    """
    from ultralytics import YOLO
    yolo   = YOLO(YOLO_WEIGHTS)
    model  = yolo.model.to(device).eval()

    # The Classify head is model.model[-1].
    # Hook into model.model[-2] — the last spatial feature layer.
    target_layer = model.model[-2]
    gcam = GradCAM(target_layer)

    inp = tensor.unsqueeze(0).requires_grad_(True)
    logits = model(inp)
    if isinstance(logits, (list, tuple)):
        logits = logits[-1]

    pred_idx   = logits.argmax(dim=1).item()
    confidence = F.softmax(logits, dim=1)[0, pred_idx].item()

    model.zero_grad()
    logits[0, pred_idx].backward()

    cam = gcam.compute()
    label = class_names.get(pred_idx, f"class_{pred_idx}")
    return cam, label, confidence


# ── Model 2: EfficientNet-B0 ──────────────────────────────────────────────────

def _build_efficientnet_b0(num_classes: int):
    """Inlined to avoid import collision with franken/model.py."""
    import torch.nn as nn
    from torchvision import models
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(m.classifier[1].in_features, num_classes),
    )
    return m


def run_effnet_gradcam(tensor: torch.Tensor, class_names: dict, device: str):
    """
    Hook into features[-1] (last MBConv block, output shape (B, 1280, 7, 7)).
    """
    model = _build_efficientnet_b0(num_classes=len(class_names)).to(device).eval()
    model.load_state_dict(torch.load(EFFNET_WEIGHTS, map_location=device))

    gcam = GradCAM(model.features[-1])

    inp = tensor.unsqueeze(0).requires_grad_(True)
    logits = model(inp)

    pred_idx   = logits.argmax(dim=1).item()
    confidence = F.softmax(logits, dim=1)[0, pred_idx].item()

    model.zero_grad()
    logits[0, pred_idx].backward()

    cam   = gcam.compute()
    label = class_names.get(pred_idx, f"class_{pred_idx}")
    return cam, label, confidence


# ── Model 3: TaxonomicYOLO26 (two passes) ────────────────────────────────────

def _franken_forward_with_hook(model, tensor, head: str, genera_names: dict, species_names: dict):
    """
    Single forward+backward pass for TaxonomicYOLO26.
    head: "genus" or "species"
    Hooks into _inner.model[-2] (last spatial layer before flatten).
    """
    # Hook into the last spatial backbone layer
    target_layer = model._inner.model[-2]
    gcam = GradCAM(target_layer)

    inp = tensor.unsqueeze(0).requires_grad_(True)
    genus_logits, species_logits = model(inp)

    if head == "genus":
        logits    = genus_logits
        names_map = genera_names
    else:
        logits    = species_logits
        names_map = species_names

    pred_idx   = logits.argmax(dim=1).item()
    confidence = F.softmax(logits, dim=1)[0, pred_idx].item()

    model.zero_grad()
    logits[0, pred_idx].backward(retain_graph=(head == "genus"))

    cam   = gcam.compute()
    label = names_map.get(pred_idx, f"class_{pred_idx}")
    return cam, label, confidence


def run_franken_gradcam(tensor: torch.Tensor, class_names: dict, device: str):
    """Returns (genus_cam, genus_label, genus_conf, species_cam, species_label, species_conf)."""
    from model import TaxonomicYOLO26

    ckpt  = torch.load(FRANKEN_WEIGHTS, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    # Infer num_genera from the saved genus head
    num_genera  = state["genus_head.1.weight"].shape[0]
    num_species = len(class_names)

    # Load genus names if saved alongside the checkpoint
    if os.path.exists(FRANKEN_GENERA):
        with open(FRANKEN_GENERA) as f:
            genera_list  = json.load(f)
        genera_names = {i: n for i, n in enumerate(genera_list)}
    else:
        genera_names = {i: f"genus_{i}" for i in range(num_genera)}

    model = TaxonomicYOLO26(
        num_genera=num_genera,
        num_species=num_species,
        weights=YOLO_WEIGHTS,
    ).to(device).eval()

    # Filter out keys whose shape doesn't match (e.g. stock 1000-class inner head)
    model_state  = model.state_dict()
    filtered     = {k: v for k, v in state.items()
                    if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(filtered, strict=False)

    # Two separate passes so gradients don't interfere
    genus_cam, genus_label, genus_conf       = _franken_forward_with_hook(
        model, tensor, "genus",   genera_names, class_names)
    species_cam, species_label, species_conf = _franken_forward_with_hook(
        model, tensor, "species", genera_names, class_names)

    return genus_cam, genus_label, genus_conf, species_cam, species_label, species_conf


# ── Plotting ───────────────────────────────────────────────────────────────────

def build_figure(img_np, results: list, out_path: str):
    """
    results: list of (title, cam, label, confidence)
    Produces N rows × 3 cols (Original | Heatmap | Overlay).
    """
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(14, n * 4))
    fig.patch.set_facecolor("#0f172a")

    if n == 1:
        axes = [axes]

    # Column headers on first row
    for ax, col in zip(axes[0], ["Original", "Grad-CAM Heatmap", "Overlay + Prediction"]):
        ax.set_title(col, color="white", fontsize=11, fontweight="bold", pad=8)

    for row, (row_title, cam, label, conf) in enumerate(results):
        heatmap    = plt.cm.jet(cam)[:, :, :3]
        overlay_np = overlay(img_np, cam)

        axes[row][0].imshow(img_np)
        axes[row][1].imshow(heatmap)
        axes[row][2].imshow(overlay_np)
        axes[row][2].set_title(
            f"{label}\n{conf*100:.1f}% confidence",
            color="#86efac", fontsize=9, pad=6
        )

        # Row label on the left
        axes[row][0].set_ylabel(row_title, color="white", fontsize=10,
                                fontweight="bold", rotation=90, labelpad=8)

        for ax in axes[row]:
            ax.axis("off")
            ax.set_facecolor("#0f172a")
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.suptitle("Model Comparison — Grad-CAM Visualisation\nWhat does each model focus on?",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  >> Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GradCAM comparison across all three models")
    parser.add_argument("--image", required=True, help="Path to input mushroom image")
    parser.add_argument("--device", default=None, help="cuda or cpu (auto-detected if omitted)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Image  : {args.image}")

    # Load class names
    with open(CLASS_NAMES_JSON) as f:
        raw = json.load(f)
    class_names = {int(k): v for k, v in raw.items()}
    print(f"Classes: {len(class_names)}")

    tensor, pil_img = load_image(args.image, device)
    img_np = pil_to_np(pil_img)

    results = []

    # ── YOLOv26n-cls ──────────────────────────────────────────────────────────
    print("\n[1/4] Running YOLOv26n-cls GradCAM...")
    try:
        cam, label, conf = run_yolo_gradcam(tensor.clone(), class_names, device)
        results.append(("YOLOv26n-cls\n(Production model)", cam, label, conf))
        print(f"      → {label}  {conf*100:.1f}%")
    except Exception as e:
        print(f"      ✗ Failed: {e}")

    # ── EfficientNet-B0 ───────────────────────────────────────────────────────
    print("\n[2/4] Running EfficientNet-B0 GradCAM...")
    try:
        cam, label, conf = run_effnet_gradcam(tensor.clone(), class_names, device)
        results.append(("EfficientNet-B0\n(ImageNet pretrained)", cam, label, conf))
        print(f"      → {label}  {conf*100:.1f}%")
    except Exception as e:
        print(f"      ✗ Failed: {e}")

    # ── TaxonomicYOLO26 (genus head) ──────────────────────────────────────────
    print("\n[3/4] Running TaxonomicYOLO26 GradCAM (genus head)...")
    try:
        g_cam, g_label, g_conf, s_cam, s_label, s_conf = run_franken_gradcam(
            tensor.clone(), class_names, device)
        results.append(("TaxonomicYOLO26\n— Genus head", g_cam, g_label, g_conf))
        results.append(("TaxonomicYOLO26\n— Species head", s_cam, s_label, s_conf))
        print(f"      Genus   → {g_label}  {g_conf*100:.1f}%")
        print(f"      Species → {s_label}  {s_conf*100:.1f}%")
    except Exception as e:
        print(f"      ✗ Failed: {e}")

    if not results:
        print("\nAll models failed — check weights paths and dependencies.")
        return

    print("\n[4/4] Building figure...")
    build_figure(img_np, results, OUT_PATH)
    print(f"\nDone. Open: {OUT_PATH}")


if __name__ == "__main__":
    main()
