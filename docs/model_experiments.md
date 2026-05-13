# New Model Experiments

This folder explores two architectures that were **not tested anywhere else in this project**.
The goal is to find out whether either can beat the current production model (YOLOv26n-cls, 88.1% top-1)
or the best-accuracy model (EfficientNet-B0, 89.5% top-1).

Both models are trained on the exact same 169-species dataset, same splits, same augmentation
pipeline, and same loss functions as `scripts/training/cnn/` — so results are directly comparable.

---

## Context — What Already Existed

Before this folder was created, the main project had trained and compared four models:

| Model | Params | Top-1 | Top-5 | Size | Role |
|---|---|---|---|---|---|
| YOLOv8n-cls | 2.7M | 86.8% | 97.9% | 3.4 MB | Original baseline |
| **YOLOv26n-cls** | **1.74M** | **88.1%** | **98.4%** | **3.6 MB** | **Production model** |
| EfficientNet-B0 | 5.3M | 89.5% | 97.9% | 17.2 MB | Highest top-1 |
| TaxonomicYOLO26 | ~2M | 79.7% | — | 25.0 MB | Dual-head genus+species |

Two architecture families were completely absent from this comparison:
- **Attention-based models** (Vision Transformers)
- **Modern CNNs** that incorporate ViT design ideas (ConvNeXt)

This folder fills both gaps.

---

## What Was Built

### Folder structure

```
Herman/
├── README.md               ← this file
├── compare.py              ← ranked leaderboard across ALL project models
├── shared/
│   ├── dataset.py          ← shared dataloader (reused by both models)
│   └── losses.py           ← shared loss functions and optimisers
├── vit/
│   ├── model.py            ← ViT-S/16 architecture (via timm)
│   ├── train.py            ← training script with warmup + split LRs
│   └── evaluate.py         ← test-set evaluation, confusion matrix, per-class accuracy
└── convnext/
    ├── model.py            ← ConvNeXt-Tiny architecture (via torchvision)
    ├── train.py            ← training script (same pattern as EfficientNet baseline)
    └── evaluate.py         ← test-set evaluation, confusion matrix, per-class accuracy
```

### `shared/dataset.py`
A copy of the dataloader from `scripts/training/cnn/dataset.py`. Both ViT-S/16 and
ConvNeXt-Tiny use 224×224 ImageNet-normalised inputs, so no changes were needed.
Keeping it in `shared/` means both models use identical augmentation — there's no
variable that could explain away differences in results.

Augmentation used during training:
- `RandomResizedCrop(224, scale=0.6–1.0)` — varied zoom and framing
- `RandomHorizontalFlip` — free augmentation (mushrooms are left-right symmetric)
- `ColorJitter(brightness, contrast, saturation, hue)` — lighting variance
- `RandomRotation(15°)` — any angle in the field

### `shared/losses.py`
Identical to `scripts/training/cnn/losses.py`. Provides:
- `CrossEntropyLoss` (hard labels)
- `CrossEntropyLoss` with label smoothing 0.1
- `FocalLoss` (gamma=2) — down-weights easy samples, useful for rare species
- `FocalLoss` + label smoothing combined
- Optimiser factory: AdamW, SGD (Nesterov), RAdam

---

## Model 1 — ViT-S/16 (Vision Transformer Small, patch size 16)

### Why ViT?
Every model trained in this project so far uses convolutions. Convolutions are inherently
**local** — they look at small patches and gradually build up global understanding through
stacking layers. Vision Transformers take the opposite approach: every image patch attends
to every other patch from the very first layer. For fine-grained classification tasks
(like distinguishing 169 mushroom species by subtle features such as gill structure, cap
texture, and stem shape), that global attention is a meaningful advantage.

The question this experiment answers: *does global self-attention help on this dataset,
or do the inductive biases of CNNs matter more given the domain?*

### Architecture
- Splits each 224×224 image into 196 patches of 16×16 pixels
- Each patch becomes a token; positional embeddings encode spatial location
- 12 transformer blocks, each with multi-head self-attention + MLP
- Classification token (`[CLS]`) at the end → Linear(384 → 169)
- Pretrained on **ImageNet-21k** (then fine-tuned on ImageNet-1k) via `timm`
- ~22M parameters

### Key training differences vs EfficientNet
ViTs are more sensitive to learning rate than CNNs. Three specific changes were made:

| Setting | EfficientNet-B0 | ViT-S/16 | Reason |
|---|---|---|---|
| Base LR | 1e-3 | 1e-4 | Higher LR causes ViT divergence |
| Warmup | None | 5 epochs linear | Randomly-initialised head creates large early gradients |
| Backbone LR | Same as head | 10% of head LR | Prevents catastrophic forgetting of pretrained features |
| Weight decay | 1e-4 | 0.05 | Standard for ViT AdamW fine-tuning |

### Files
| File | Purpose |
|---|---|
| `vit/model.py` | `build_vit_small(num_classes)` — one-liner via `timm.create_model` |
| `vit/train.py` | Full training loop with warmup scheduler + parameter group split |
| `vit/evaluate.py` | Test-set top-1/top-5, confusion matrix PNG, top_errors.txt, per_class_accuracy.txt |

---

## Model 2 — ConvNeXt-Tiny

### Why ConvNeXt?
EfficientNet-B0 currently holds the best top-1 (89.5%) in this project. ConvNeXt (Liu et al.,
"A ConvNet for the 2020s", 2022) was designed by asking: *what if we took a standard ResNet
and systematically applied every design lesson learned from Vision Transformers?*

Changes from a standard CNN:
- **7×7 depthwise convolutions** (vs 3×3 in ResNet) — larger receptive field per layer
- **Inverted bottleneck** — expand channels wide in the middle, like ViT's FFN
- **LayerNorm** instead of BatchNorm — more stable, better transfer learning
- **GELU** instead of ReLU
- Fewer normalisation/activation layers per block — cleaner gradient flow

The result is a pure CNN that matches ViT on ImageNet benchmarks while being faster
to train and more memory-efficient. This experiment tests whether those gains transfer
to mushroom classification.

### Architecture
- 4 stages of depthwise-separable convolution blocks
- Feature map: 768 channels at the bottleneck
- Head: `LayerNorm → Flatten → Dropout(0.3) → Linear(768 → 169)`
- Pretrained on **ImageNet-1k** via `torchvision`
- ~28M parameters

### Key training differences vs EfficientNet
ConvNeXt behaves similarly to EfficientNet — no warmup needed, same cosine schedule.
The only change is a slightly lower LR (`4e-4` vs `1e-3`) to account for the larger
model capacity.

### Files
| File | Purpose |
|---|---|
| `convnext/model.py` | `build_convnext_tiny(num_classes)` — torchvision model with custom head |
| `convnext/train.py` | Training loop — identical pattern to `scripts/training/cnn/train.py` |
| `convnext/evaluate.py` | Test-set top-1/top-5, confusion matrix PNG, top_errors.txt, per_class_accuracy.txt |

---

## How to Run

### 1. Install dependency (ViT only — one time)
```bash
pip install timm
```

### 2. Train
```bash
# From the project root:
python Herman/vit/train.py
python Herman/convnext/train.py

# With explicit options (defaults shown):
python Herman/vit/train.py      --optimizer adamw --loss ce_smooth
python Herman/convnext/train.py --optimizer adamw --loss ce_smooth

# Other optimizer options: sgd | radam
# Other loss options:      ce | focal | focal_smooth
```

### 3. Evaluate on held-out test set
```bash
python Herman/vit/evaluate.py      --run vit_small_adamw_ce_smooth
python Herman/convnext/evaluate.py --run convnext_tiny_adamw_ce_smooth
```

### 4. Compare everything
```bash
python Herman/compare.py
```
This scans `docs/cnn_runs/` and `docs/convnext_runs/` and prints a single ranked
leaderboard including the YOLO models, EfficientNet, ViT, and ConvNeXt.

---

## Output Locations

All training outputs are saved under `docs/convnext_runs/<run_name>/`:

```
docs/convnext_runs/
├── vit_small_adamw_ce_smooth/
│   ├── weights/
│   │   ├── best.pt          ← best validation top-1 checkpoint
│   │   └── last.pt          ← final epoch checkpoint
│   ├── results.csv          ← per-epoch metrics log
│   ├── confusion_matrix_normalized.png
│   ├── top_errors.txt       ← 20 most-confused species pairs
│   └── per_class_accuracy.txt
└── convnext_tiny_adamw_ce_smooth/
    └── (same structure)
```

---

## Results

> Training has not been run yet. Fill in this table after each run and update `docs/model_comparison.md`.

### Full project leaderboard (run `python Herman/compare.py` to generate live)

| Rank | Model | Top-1 | Top-5 | Params | Size | Notes |
|---|---|---|---|---|---|---|
| — | EfficientNet-B0 | 89.5% | 97.9% | 5.3M | 17.2 MB | Existing best accuracy |
| — | YOLOv26n-cls | 88.1% | 98.4% | 1.74M | 3.6 MB | Production model |
| — | YOLOv8n-cls | 86.8% | 97.9% | 2.7M | 3.4 MB | Baseline |
| — | **ConvNeXt-Tiny** | **[TBD]** | **[TBD]** | 28M | ~110 MB | Herman experiment |
| — | **ViT-S/16** | **[TBD]** | **[TBD]** | 22M | ~85 MB | Herman experiment |

### Hypothesis (before training)

| Model | Expected Top-1 | Reasoning |
|---|---|---|
| ConvNeXt-Tiny | ~90–92% | Consistently beats EfficientNet-B0 on ImageNet at similar scale |
| ViT-S/16 | ~88–92% | Strong for fine-grained tasks; 689k images is enough to fine-tune well |

### Training configuration reference

| Setting | ViT-S/16 | ConvNeXt-Tiny | EfficientNet-B0 (reference) |
|---|---|---|---|
| Base LR | 1e-4 | 4e-4 | 1e-3 |
| Backbone LR | 1e-5 (0.1×) | — (unified) | — (unified) |
| Warmup | 5 epochs linear | None | None |
| LR schedule | Warmup → Cosine | Cosine | Cosine |
| Weight decay | 0.05 | 1e-4 | 1e-4 |
| Batch size | 32 | 32 | 32 |
| Max epochs | 50 | 50 | 50 |
| Early stopping | 10 epochs | 10 epochs | 10 epochs |
| Grad clip | 1.0 | 1.0 | 1.0 |
| Default loss | ce_smooth | ce_smooth | ce_smooth |
| Pretrain data | ImageNet-21k | ImageNet-1k | ImageNet-1k |
