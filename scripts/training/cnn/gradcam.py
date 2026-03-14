"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for EfficientNet-B0.

What is Grad-CAM?
-----------------
Grad-CAM answers the question "which pixels caused this prediction?"
It works by:
  1. Running a forward pass to get a prediction.
  2. Running a backward pass for the predicted class score.
  3. Taking the gradients of that score w.r.t. the LAST convolutional layer's
     feature maps — large gradients mean those spatial locations mattered.
  4. Global-average-pooling those gradients into per-channel weights.
  5. Computing a weighted sum of the feature maps → a coarse 7×7 heatmap.
  6. Upsampling to 224×224 and overlaying on the original image.

Why is this useful?
-------------------------------------
  - It shows the model is looking at the mushroom cap/stem, not the background.
  - It reveals failure modes: if the heatmap lights up the background, the model
    learned spurious correlations from the dataset, not actual mushroom features.
  - It's the standard explainability tool cited in ML papers.

Usage (from project root):
    # Sample 16 random test images from the best run:
    python scripts/training/cnn/gradcam.py --run efficientnet_b0_adamw_focal_smooth

    # Run on a specific image file:
    python scripts/training/cnn/gradcam.py --run efficientnet_b0_adamw_focal_smooth \\
        --image path/to/mushroom.jpg

    # Control how many test samples to visualise (default 16):
    python scripts/training/cnn/gradcam.py --run efficientnet_b0_adamw_focal_smooth --n 32

Output:
    docs/cnn_runs/<run>/gradcam/
        gradcam_grid.png     — grid of (original | heatmap | overlay) for each sample
        gradcam_single_*.png — individual full-size images (one per sample)
"""

import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from dataset import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from model import build_efficientnet_b0

# ─── Paths ────────────────────────────────────────────────────────────────────

_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DATA_DIR = os.path.join(_ROOT, "data", "dataset_split")


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Hooks into the last convolutional block of EfficientNet-B0 to capture
    both the forward activations and the backward gradients for a given class.

    Target layer: model.features[-1]
        This is the final MBConv block before the adaptive average pool.
        Its output shape is (batch, 1280, 7, 7) for a 224×224 input — small
        enough to average over, rich enough to localise the subject.
    """

    def __init__(self, model: torch.nn.Module):
        self.model       = model
        self.activations = None
        self.gradients   = None

        # The last block in EfficientNet-B0's feature extractor
        target_layer = model.features[-1]

        # Forward hook — called every time the layer produces output.
        # We detach() so the stored tensor doesn't hold graph memory.
        target_layer.register_forward_hook(self._save_activation)

        # Full-backward hook — called during loss.backward().
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _input, output):
        self.activations = output.detach()          # (1, C, H, W)

    def _save_gradient(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0].detach()    # (1, C, H, W)

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Returns:
            cam     (np.ndarray) normalised heatmap, shape (224, 224), values [0, 1]
            pred_idx (int)       predicted class index
            confidence (float)   softmax probability of predicted class
        """
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0)    # add batch dim: (1, 3, 224, 224)
        input_tensor.requires_grad_(True)

        # ── Forward ───────────────────────────────────────────────────────────
        logits = self.model(input_tensor)
        probs  = F.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        confidence = probs[0, class_idx].item()

        # ── Backward for target class ─────────────────────────────────────────
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # ── Compute CAM ───────────────────────────────────────────────────────
        # Weight each channel of the activation map by its mean gradient.
        # Large positive gradient → this channel's activations support the class.
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)     # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # ReLU: we only care about features that contribute *positively*
        cam = F.relu(cam)

        # Upsample the 7×7 CAM to match the input image size (224×224)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx, confidence


# ─── Image utilities ──────────────────────────────────────────────────────────

def load_and_preprocess(image_path: str, device: str):
    """Load an image, apply the same eval transform used during training."""
    eval_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img   = Image.open(image_path).convert("RGB")
    img   = transforms.Resize(256)(img)
    img   = transforms.CenterCrop(IMG_SIZE)(img)
    tensor = eval_tfm(Image.open(image_path).convert("RGB")).to(device)
    return tensor, img


def tensor_to_display(tensor: torch.Tensor):
    """Undo ImageNet normalisation so we can display the image."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    img  = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


def overlay_heatmap(img_np: np.ndarray, cam: np.ndarray, alpha: float = 0.45):
    """
    Blend the Grad-CAM heatmap (jet colormap) onto the original image.
    alpha controls heatmap opacity — 0.45 gives a clear overlay without
    obscuring the original texture.
    """
    heatmap = plt.cm.jet(cam)[:, :, :3]    # jet colourmap, drop alpha channel
    return np.clip(img_np * (1 - alpha) + heatmap * alpha, 0, 1)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_grid(samples, save_dir: str, class_names: list):
    """
    Save a grid of (original | heatmap | overlay | label) panels.
    Each row is one sample.
    """
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 3.5))

    if n == 1:
        axes = [axes]   # make iterable

    col_titles = ["Original", "Grad-CAM Heatmap", "Overlay + Prediction"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

    for row_idx, (img_np, cam, pred_idx, confidence, true_label) in enumerate(samples):
        heatmap = plt.cm.jet(cam)[:, :, :3]
        overlay = overlay_heatmap(img_np, cam)

        axes[row_idx][0].imshow(img_np)
        axes[row_idx][1].imshow(heatmap)
        axes[row_idx][2].imshow(overlay)

        pred_label = class_names[pred_idx]
        correct    = "✓" if pred_label == true_label else "✗"
        title      = f"{correct} Pred: {pred_label}\n({confidence*100:.1f}%)"
        if pred_label != true_label:
            title += f"\nTrue: {true_label}"
        axes[row_idx][2].set_title(title, fontsize=8, color="green" if correct == "✓" else "red")

        for ax in axes[row_idx]:
            ax.axis("off")

    fig.suptitle("Grad-CAM — EfficientNet-B0 Mushroom Classifier", fontsize=13, y=1.01)
    plt.tight_layout()

    path = os.path.join(save_dir, "gradcam_grid.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  >> Grid saved: {path}")


def plot_single(img_np, cam, pred_idx, confidence, true_label, class_names, save_path):
    """Full-size 3-panel figure for one sample."""
    heatmap = plt.cm.jet(cam)[:, :, :3]
    overlay = overlay_heatmap(img_np, cam)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np);   axes[0].set_title("Original",          fontsize=12)
    axes[1].imshow(heatmap);  axes[1].set_title("Grad-CAM Heatmap",  fontsize=12)
    axes[2].imshow(overlay)

    pred_label = class_names[pred_idx]
    correct    = "✓ Correct" if pred_label == true_label else "✗ Wrong"
    color      = "green"     if pred_label == true_label else "red"
    axes[2].set_title(
        f"Overlay\n{correct}: {pred_label} ({confidence*100:.1f}%)\nTrue: {true_label}",
        fontsize=10, color=color,
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Sample collection ────────────────────────────────────────────────────────

def collect_test_samples(test_dir: str, n: int):
    """
    Randomly pick n image paths from the test split.
    Returns list of (image_path, true_class_name).
    """
    samples = []
    class_dirs = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    ])

    all_images = []
    for cls in class_dirs:
        cls_dir = os.path.join(test_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                all_images.append((os.path.join(cls_dir, fname), cls))

    random.shuffle(all_images)
    return all_images[:n]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualisations")
    parser.add_argument("--run", required=True,
                        help="Run folder name inside docs/cnn_runs/")
    parser.add_argument("--image", default=None,
                        help="Path to a single image file (overrides --n)")
    parser.add_argument("--n", type=int, default=16,
                        help="Number of random test images to visualise (default 16)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for test image sampling (default 42)")
    args = parser.parse_args()

    run_dir  = os.path.join(_ROOT, "docs", "cnn_runs", args.run)
    best_pt  = os.path.join(run_dir, "weights", "best.pt")
    save_dir = os.path.join(run_dir, "gradcam")

    print("=========================================")
    print("  Grad-CAM  |  EfficientNet-B0")
    print(f"  Run: {args.run}")
    print("=========================================")

    if not os.path.exists(best_pt):
        print(f"ERROR: No weights at {best_pt} — run train.py first.")
        return

    os.makedirs(save_dir, exist_ok=True)

    # ── Class names from the dataset folder structure ──────────────────────
    test_dir    = os.path.join(DATA_DIR, "test")
    class_names = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    ])
    num_classes = len(class_names)
    print(f">> Classes: {num_classes}")

    # ── Model + Grad-CAM hook ──────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">> Device: {device}")

    model = build_efficientnet_b0(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(best_pt, map_location=device))
    gradcam = GradCAM(model)

    # ── Collect images ─────────────────────────────────────────────────────
    if args.image:
        raw_samples = [(args.image, os.path.basename(os.path.dirname(args.image)))]
    else:
        random.seed(args.seed)
        raw_samples = collect_test_samples(test_dir, args.n)
        print(f">> Sampled {len(raw_samples)} images from test set")

    # ── Run Grad-CAM on each image ─────────────────────────────────────────
    processed = []
    for i, (img_path, true_label) in enumerate(raw_samples):
        tensor, pil_img = load_and_preprocess(img_path, device)
        cam, pred_idx, confidence = gradcam.generate(tensor)

        img_np = np.array(pil_img).astype(np.float32) / 255.0
        processed.append((img_np, cam, pred_idx, confidence, true_label))

        # Save individual full-size figure
        single_path = os.path.join(save_dir, f"gradcam_single_{i:03d}.png")
        plot_single(img_np, cam, pred_idx, confidence, true_label, class_names, single_path)

        pred_label = class_names[pred_idx]
        correct    = "✓" if pred_label == true_label else "✗"
        print(f"  [{i+1:>3}/{len(raw_samples)}] {correct} {pred_label} ({confidence*100:.1f}%) — true: {true_label}")

    # ── Grid figure ────────────────────────────────────────────────────────
    plot_grid(processed, save_dir, class_names)

    correct_count = sum(1 for _, _, p, _, t in processed if class_names[p] == t)
    print(f"\nAccuracy on this sample: {correct_count}/{len(processed)} ({100*correct_count/len(processed):.1f}%)")
    print(f"All Grad-CAM outputs saved to: {save_dir}")


if __name__ == "__main__":
    main()
