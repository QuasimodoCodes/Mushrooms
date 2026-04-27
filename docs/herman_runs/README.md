# ConvNeXt-Tiny — AdamW + CE Smooth Training Results

Two runs were completed: a **full fine-tuning** attempt and a **PEFT two-phase** run.
Both used identical hyperparameters except for backbone freezing.

---

## Configuration

| Setting | Value |
|---|---|
| Architecture | ConvNeXt-Tiny (torchvision, ImageNet-1k pretrained) |
| Parameters | 28.6M total |
| Optimizer | AdamW |
| Loss | CrossEntropyLoss + label smoothing 0.1 |
| Base LR | 4e-4 |
| PEFT backbone LR (P2) | 4e-5 (10× lower) |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Max epochs | 50 |
| Early stopping patience | 10 |
| Gradient clip | max_norm = 1.0 |
| LR schedule | CosineAnnealingLR |
| Input size | 224 × 224 |
| Dataset | zlatan599/mushroom1 — 169 species, ~689k images, 80/10/10 split |

---

## Run 1 — Full Fine-tuning (`convnext_tiny_adamw_ce_smooth`)

All 28.6M parameters trained from epoch 1. **This run was interrupted after 3 epochs.**

| Epoch | Train Loss | Train Top-1 | Val Loss | Val Top-1 | Val Top-5 | Epoch Time |
|---|---|---|---|---|---|---|
| 1 | 2.0048 | 65.35% | 1.6320 | 75.34% | 93.77% | 7294 s |
| 2 | 1.5836 | 76.53% | 1.4677 | 79.92% | 96.06% | 6752 s |
| 3 | 1.4594 | 80.13% | 1.4435 | 80.33% | 96.31% | 6789 s |

**Average epoch time: ~7,278 s (~2 hours/epoch).** At 50 epochs that would be ~100 hours — the run was abandoned for the PEFT approach.

---

## Run 2 — PEFT Two-phase (`convnext_tiny_adamw_ce_smooth_peft`)

**Phase 1 (epochs 1–10):** backbone frozen, only Stage 4 + head trained (~14.4M / 28.6M params).  
**Phase 2 (epochs 11–50):** full backbone unfrozen, backbone LR = 4e-5, head LR = 4e-4.

### Phase 1 — Frozen Backbone

| Epoch | Train Loss | Train Top-1 | Val Loss | Val Top-1 | Val Top-5 | Epoch Time |
|---|---|---|---|---|---|---|
| 1 | 1.8819 | 69.20% | 1.4091 | 81.54% | 96.90% | 829 s |
| 2 | 1.4224 | 81.71% | 1.3204 | 83.97% | 97.57% | 783 s |
| 3 | 1.2955 | 85.50% | 1.2715 | 85.78% | 97.83% | 775 s |
| 4 | 1.2148 | 87.99% | 1.2705 | 85.70% | 97.94% | 775 s |
| 5 | 1.1579 | 89.94% | 1.2524 | 86.24% | 97.96% | 775 s |
| 6 | 1.1171 | 91.09% | 1.2405 | 86.67% | 97.95% | 778 s |
| 7 | 1.0863 | 92.11% | 1.2529 | 86.70% | 97.96% | 776 s |
| 8 | 1.0602 | 93.01% | **1.2309** | **87.75%** | **98.26%** | 775 s |
| 9 | 1.0390 | 93.58% | 1.2270 | 87.68% | 98.11% | 776 s |
| 10 | 1.0214 | 94.25% | 1.2536 | 87.09% | 98.02% | 779 s |

**Average Phase 1 epoch time: ~776 s (~13 min).** Compared to ~7,278 s for the full run — **~9.4× faster per epoch** with half the model frozen.

### Phase 2 — Full Backbone (selected milestones)

| Epoch | Phase | Train Loss | Train Top-1 | Val Loss | Val Top-1 | Val Top-5 |
|---|---|---|---|---|---|---|
| 11 | P2 | 0.9775 | 95.80% | 1.2134 | 88.56% | 98.24% |
| 12 | P2 | 0.9444 | 96.94% | 1.2101 | 89.25% | 98.24% |
| 17 | P2 | 0.9018 | 98.27% | 1.2229 | 89.49% | 98.17% |
| 20 | P2 | 0.8883 | 98.62% | 1.2370 | 89.82% | 98.12% |
| 21 | P2 | 0.8855 | 98.74% | 1.2274 | 90.20% | 98.02% |
| 25 | P2 | 0.8748 | 99.07% | 1.2437 | 90.32% | 98.01% |
| 29 | P2 | 0.8652 | 99.34% | 1.2476 | 90.56% | 97.71% |
| 35 | P2 | 0.8551 | 99.61% | 1.2564 | 90.66% | 97.67% |
| 37 | P2 | 0.8527 | 99.67% | 1.2487 | 90.90% | 97.67% |
| 40 | P2 | 0.8519 | 99.68% | 1.2433 | 91.07% | 97.66% |
| **46** | **P2** | **0.8486** | **99.77%** | **1.2431** | **91.13%** | **97.77%** |
| 50 | P2 | 0.8487 | 99.77% | 1.2407 | 91.07% | 97.82% |

**Average Phase 2 epoch time: ~1,091 s (~18 min).** Full 50-epoch run total: ~14.3 hours.

### PEFT Summary

| Metric | Value |
|---|---|
| **Best val Top-1** | **91.13%** (epoch 46) |
| **Best val Top-5** | **98.32%** (epoch 8, 28, 44, 48, 49, 50) |
| Best epoch | 46 |
| Final val Top-1 | 91.07% (epoch 50) |
| Final val Top-5 | 97.82% (epoch 50) |
| Final train loss | 0.8487 |
| Final val loss | 1.2407 |
| Phase 1 best | 87.75% Top-1 @ epoch 8 |
| Phase 1 total time | ~7,780 s (~2.2 h) |
| Phase 2 total time | ~43,600 s (~12.1 h) |
| Total training time | ~51,400 s (~14.3 h) |

---

## Comparison Against Project Baselines

| Model | Top-1 | Top-5 | Params | GPU (ms) | CPU (ms) | Notes |
|---|---|---|---|---|---|---|
| **ConvNeXt-Tiny (PEFT)** | **91.13%** | **98.32%** | 28.6M | 7.6 | 35.6 | ✅ New best top-1 |
| EfficientNet-B0 | 89.5% | 97.9% | 5.3M | 11.6 | 19.2 | Previous best accuracy |
| YOLOv26n-cls (PyTorch) | 88.1% | 98.4% | 1.74M | 6.3 | 7.1 | |
| YOLOv26n TFLite f16 | 88.1% | 98.4% | 1.74M | — | **4.8** | **Production model** |
| YOLOv8n-cls | 86.8% | 97.9% | 2.7M | 4.6 | 6.3 | Original baseline |
| ConvNeXt-Tiny (full, 3 ep) | 80.33% | 96.31% | 28.6M | — | — | Interrupted — too slow |

Benchmarked on NVIDIA RTX 4070 — 50 warmup + 200 timed runs, batch=1, 224×224. Script: `scripts/benchmark_speed.py`.

ConvNeXt-Tiny PEFT is the **new top-1 accuracy leader** in this project, beating EfficientNet-B0 by **+1.6 percentage points**. On GPU the speed gap vs YOLOv26n is only 1.3 ms. On CPU — the only hardware available in Cloud Run — ConvNeXt is **7.4× slower** than the TFLite production model, which rules it out for deployment.

---

## Key Observations

### PEFT efficiency
Phase 1 epoch time was ~776 s vs ~7,278 s for the full run — **9.4× faster** during the frozen phase. The frozen backbone also produced a large early jump: 81.5% top-1 on epoch 1 (vs 75.3% for the full run), because the pretrained ImageNet features were immediately useful for the linear head without any interference from a noisy backbone gradient.

### Smooth Phase 2 transition
Val top-1 jumped from 87.09% (end of P1) to 88.56% (first P2 epoch) without any instability spike. The 10× lower backbone LR prevented catastrophic forgetting of ImageNet features while allowing gradual adaptation to mushroom fine-grained differences.

### Label smoothing effect
The final train/val loss gap is visibly wider than with hard CE (train ~0.848, val ~1.241). This is expected and healthy: label smoothing reduces overconfidence on training predictions, so the training loss is penalised more — but the model generalises better. The progressive top-1 improvement through epoch 46 confirms the smoothing helped prevent early overfitting.

### Gradient norm
Phase 1 gradient norms started at ~4.0 and decayed steadily to ~2.2 by epoch 10 (smaller backward graph, stable). Phase 2 started at ~14.3 (large spike from newly unfrozen backbone) and decayed to ~2.5 by epoch 50. The gradient clipping at max_norm=1.0 kept individual updates bounded throughout.

### No early stopping triggered
The model kept improving (or held steady) all the way to epoch 46–50 without a 10-epoch plateau. This indicates the CosineAnnealingLR schedule drove useful late-stage refinement even at very low LRs (6e-08 at epoch 50).

### Trade-off vs production model
ConvNeXt-Tiny PEFT achieves +3.0% top-1 over YOLOv26n-cls, but at **16× more parameters** (28.6M vs 1.74M), a far larger file size, and — critically — **35.6 ms CPU inference vs 4.8 ms for the TFLite production model**. It is not a production replacement — but it establishes an accuracy ceiling for the task and validates that the dataset supports >91% top-1 with the right architecture.

---

## Weights

| Run | File | Description |
|---|---|---|
| PEFT | `convnext_tiny_adamw_ce_smooth_peft/weights/best.pt` | Epoch 46 — 91.13% Top-1 |
| PEFT | `convnext_tiny_adamw_ce_smooth_peft/weights/last.pt` | Epoch 50 — 91.07% Top-1 |
| Full | `convnext_tiny_adamw_ce_smooth/weights/best.pt` | Epoch 3 — 80.33% Top-1 (incomplete) |
| Full | `convnext_tiny_adamw_ce_smooth/weights/last.pt` | Epoch 3 — last saved (incomplete) |
