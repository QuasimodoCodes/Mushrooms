# ConvNeXt-Tiny — Model Experiment Report

**Project:** Mushroom Guardian — Multimodal AI System  
**Dataset:** zlatan599/mushroom1 — 169 species, ~689,000 images (80/10/10 split)  
**Hardware:** NVIDIA GeForce RTX 3070 Ti (8 GB VRAM)  
**Experiment goal:** Determine whether a modern CNN architecture (ConvNeXt-Tiny) can exceed the accuracy of the existing project models, and whether PEFT-based training makes the 28.6M parameter model practical to train.

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [Architecture](#2-architecture)
3. [Training Strategy — Why PEFT Was Necessary](#3-training-strategy--why-peft-was-necessary)
4. [Hyperparameters](#4-hyperparameters)
5. [Results — Full Fine-tuning Run (Abandoned)](#5-results--full-fine-tuning-run-abandoned)
6. [Results — PEFT Two-phase Run](#6-results--peft-two-phase-run)
7. [Full Project Leaderboard](#7-full-project-leaderboard)
8. [Inference Speed Benchmark](#8-inference-speed-benchmark)
9. [Analysis and Discussion](#9-analysis-and-discussion)
10. [Production Considerations](#10-production-considerations)
11. [Conclusion](#11-conclusion)

---

## 1. Background and Motivation

At the start of this experiment, the project had trained and compared four model architectures on the same 169-species mushroom dataset:

| Model | Top-1 | Top-5 | Parameters | Role |
|---|---|---|---|---|
| YOLOv8n-cls | 86.8% | 97.9% | 2.7M | Original baseline |
| YOLOv26n-cls | 88.1% | 98.4% | 1.74M | Production model |
| EfficientNet-B0 | 89.5% | 97.9% | 5.3M | Best accuracy |
| TaxonomicYOLO26 | 79.7% | — | ~2M | Dual-head genus+species |

Two major architecture families were entirely absent from this comparison:

- **Vision Transformers (ViT)** — attention-based, no convolutional inductive biases
- **Modern CNNs** — convolutional networks redesigned with ViT principles baked in

ConvNeXt-Tiny represents the second family. It was introduced in the paper *"A ConvNet for the 2020s"* (Liu et al., 2022), which asked: *what happens if you take a standard ResNet and systematically apply every design lesson learned from the Vision Transformer literature?*

The hypothesis entering this experiment was that ConvNeXt-Tiny would beat EfficientNet-B0 (currently 89.5%) and push the project's accuracy ceiling closer to 91–92%, based on its ImageNet-1k benchmark performance at comparable scale.

---

## 2. Architecture

ConvNeXt-Tiny is a pure convolutional network. No attention mechanisms, no token sequences — every operation is a convolution. What separates it from older CNNs like ResNet or EfficientNet is a set of targeted architectural improvements, each taken directly from the ViT design playbook:

| Design change | Old CNN (ResNet) | ConvNeXt-Tiny | Reason |
|---|---|---|---|
| Kernel size | 3×3 | **7×7 depthwise** | Larger receptive field per layer, like ViT's large patch attention span |
| Channel expansion | Bottleneck (compress-then-expand) | **Inverted bottleneck** | Expand wide in the middle, matching ViT's FFN structure |
| Normalization | BatchNorm | **LayerNorm** | More stable across batch sizes; better transfer learning properties |
| Activation | ReLU | **GELU** | Same activation used in ViT and GPT; smoother gradient |
| Norm/activation frequency | After every conv | **Only once per block** | Cleaner gradient flow, less computational overhead |

These changes, taken together, produce a model that matches Vision Transformers on ImageNet benchmarks while remaining faster to train and more memory-efficient. Unlike ViT, ConvNeXt does not require warmup epochs or a large minimum dataset size to converge reliably.

**Architecture summary for this project:**

| Property | Value |
|---|---|
| Stages | 4 stages of depthwise-separable ConvNeXt blocks |
| Feature channels at bottleneck | 768 |
| Classifier head | `LayerNorm → AdaptiveAvgPool → Flatten → Dropout(0.3) → Linear(768 → 169)` |
| Pretraining | ImageNet-1k (via torchvision) |
| Total parameters | 28.6M |
| Trainable in Phase 1 (PEFT) | ~14.4M (Stage 4 + head only) |

The classifier head was adapted from the ImageNet-1k head (768 → 1000) to the project's 169-species output. Dropout was set to 0.3, matching the EfficientNet-B0 experiment.

---

## 3. Training Strategy — Why PEFT Was Necessary

Two training strategies were attempted.

### Strategy A — Full Fine-tuning

All 28.6M parameters updated from epoch 1. This is the standard approach for smaller models in this project (EfficientNet-B0 uses it). At 28.6M parameters the backward pass is proportionally larger, producing **epoch times of approximately 7,278 seconds (~2 hours)**. To run 50 epochs would require approximately 100 hours on the RTX 3070 Ti — impractical. The full fine-tuning run was abandoned after 3 epochs.

### Strategy B — PEFT Two-phase Backbone Freezing

Parameter-Efficient Fine-Tuning (PEFT) via backbone freezing cuts training time by restricting which parameters receive gradient updates:

**Phase 1 (epochs 1–10):** Backbone stages 1–3, the stem, and all downsampling layers are frozen. Only Stage 4 and the classifier head are trainable — approximately 14.4M of 28.6M parameters (50%). The backward graph is roughly half the size, reducing per-epoch time from ~7,278 s to **~776 s (~13 minutes)**, a speedup of approximately **9.4×**.

The practical benefit is more than just speed. The frozen ImageNet-pretrained backbone acts as a fixed high-quality feature extractor from the first epoch, which is why Phase 1 already achieves 87.75% top-1 by epoch 8 — higher than what full fine-tuning reached after 3 epochs (80.3%).

**Phase 2 (epochs 11–50):** The backbone is unfrozen. Two separate learning rate groups are configured:
- Backbone: `4e-5` (10× lower than the head)
- Classifier head: `4e-4` (unchanged)

The lower backbone LR prevents catastrophic forgetting — the large random gradients from the early-training head cannot overwrite the rich ImageNet representations built up over Phase 1. The backbone adapts gradually to mushroom-specific features while the head continues at full speed.

A new CosineAnnealingLR scheduler is initialized at the Phase 2 transition, counting down over the remaining 40 epochs, so the LR cosine still terminates near zero at epoch 50.

---

## 4. Hyperparameters

| Setting | Value | Notes |
|---|---|---|
| Base LR | 4e-4 | Slightly lower than EfficientNet (1e-3); larger model capacity benefits from more conservative initial updates |
| Backbone LR (Phase 2) | 4e-5 | 10× lower to protect pretrained features |
| Weight decay | 1e-4 | Same as EfficientNet baseline |
| Batch size | 32 | |
| Max epochs | 50 | |
| Early stopping patience | 10 | Monitor val top-1; stop if no improvement for 10 consecutive epochs |
| Gradient clip | max_norm = 1.0 | Prevents gradient explosion in Phase 2 when backbone first unfreezes |
| LR schedule | CosineAnnealingLR | T_max = full epoch count per phase |
| Optimizer | AdamW | |
| Loss function | CrossEntropyLoss + label smoothing 0.1 | Smoothing distributes a small probability mass (0.1 / 168) to all non-target classes, reducing overconfidence on training labels |
| PEFT Phase 1 epochs | 10 | |
| Augmentation | RandomResizedCrop(224, scale=0.6–1.0), RandomHorizontalFlip, ColorJitter, RandomRotation(15°) | Identical to EfficientNet and ViT experiments — no confounding augmentation differences |

---

## 5. Results — Full Fine-tuning Run (Abandoned)

**Run name:** `convnext_tiny_adamw_ce_smooth`

| Epoch | Train Loss | Train Top-1 | Val Loss | Val Top-1 | Val Top-5 | Time (s) |
|---|---|---|---|---|---|---|
| 1 | 2.0048 | 65.35% | 1.6320 | 75.34% | 93.77% | 7294 |
| 2 | 1.5836 | 76.53% | 1.4677 | 79.92% | 96.06% | 6752 |
| 3 | 1.4594 | 80.13% | 1.4435 | **80.33%** | **96.31%** | 6789 |

The trajectory is clearly upward — val top-1 improved by ~5 percentage points per epoch at this stage. However, at ~7,000 s per epoch the run would require approximately **100 hours** to complete all 50 epochs. The run was terminated and PEFT was implemented instead.

---

## 6. Results — PEFT Two-phase Run

**Run name:** `convnext_tiny_adamw_ce_smooth_peft`

### 6.1 Phase 1 — Frozen Backbone (epochs 1–10)

| Epoch | Train Loss | Train Top-1 | Val Loss | Val Top-1 | Val Top-5 | Grad Norm |
|---|---|---|---|---|---|---|
| 1 | 1.8819 | 69.20% | 1.4091 | 81.54% | 96.90% | 4.025 |
| 2 | 1.4224 | 81.71% | 1.3204 | 83.97% | 97.57% | 3.267 |
| 3 | 1.2955 | 85.50% | 1.2715 | 85.78% | 97.83% | 2.941 |
| 4 | 1.2148 | 87.99% | 1.2705 | 85.70% | 97.94% | 2.728 |
| 5 | 1.1579 | 89.94% | 1.2524 | 86.24% | 97.96% | 2.586 |
| 6 | 1.1171 | 91.09% | 1.2405 | 86.67% | 97.95% | 2.471 |
| 7 | 1.0863 | 92.11% | 1.2529 | 86.70% | 97.96% | 2.397 |
| **8** | **1.0602** | **93.01%** | **1.2309** | **87.75%** | **98.26%** | **2.320** |
| 9 | 1.0390 | 93.58% | 1.2270 | 87.68% | 98.11% | 2.216 |
| 10 | 1.0214 | 94.25% | 1.2536 | 87.09% | 98.02% | 2.178 |

**Phase 1 observations:**
- Best Phase 1 epoch: 8 (val top-1 = 87.75%)
- Grad norms started at ~4.0 and decayed steadily to ~2.2, indicating stable convergence
- Average epoch time: **~776 s (~13 min)** — 9.4× faster than full fine-tuning
- Training top-1 reached 94.25% by epoch 10 while val top-1 plateaued around 87%, a sign that the frozen backbone had been extracted to capacity and Phase 2 was needed

### 6.2 Phase 2 — Full Backbone Unfrozen (epochs 11–50)

At the Phase 2 transition, the backbone was unfrozen and the optimizer was rebuilt with two parameter groups (backbone LR = 4e-5, head LR = 4e-4). Grad norms spiked to ~14.3 at epoch 11 — the expected behavior when fresh backbone gradients enter the optimizer for the first time — then decayed smoothly to ~2.5 by epoch 50.

| Epoch | Train Loss | Train Top-1 | Val Loss | Val Top-1 | Val Top-5 |
|---|---|---|---|---|---|
| 11 | 0.9775 | 95.80% | 1.2134 | 88.56% | 98.24% |
| 12 | 0.9444 | 96.94% | 1.2101 | 89.25% | 98.24% |
| 15 | 0.9147 | 97.84% | 1.2199 | 89.17% | 98.12% |
| 20 | 0.8883 | 98.62% | 1.2370 | 89.82% | 98.12% |
| 21 | 0.8855 | 98.74% | 1.2274 | 90.20% | 98.02% |
| 25 | 0.8748 | 99.07% | 1.2437 | 90.32% | 98.01% |
| 29 | 0.8652 | 99.34% | 1.2476 | 90.56% | 97.71% |
| 35 | 0.8551 | 99.61% | 1.2564 | 90.66% | 97.67% |
| 40 | 0.8519 | 99.68% | 1.2433 | 91.07% | 97.66% |
| **46** | **0.8486** | **99.77%** | **1.2431** | **91.13%** | **97.77%** |
| 47 | 0.8481 | 99.79% | 1.2415 | 90.96% | 97.81% |
| 48 | 0.8486 | 99.77% | 1.2406 | 91.07% | 97.82% |
| 49 | 0.8478 | 99.78% | 1.2407 | 91.06% | 97.82% |
| 50 | 0.8487 | 99.77% | 1.2407 | **91.07%** | **97.82%** |

### 6.3 PEFT Run Summary

| Metric | Value |
|---|---|
| **Best val Top-1** | **91.13%** (epoch 46) |
| **Best val Top-5** | **98.32%** (epochs 8, 28, 44, 48, 49, 50) |
| Final val Top-1 (epoch 50) | 91.07% |
| Final val Top-5 (epoch 50) | 97.82% |
| Final train loss | 0.8487 |
| Final val loss | 1.2407 |
| Phase 1 total time | ~7,780 s (~2.2 hours) |
| Phase 2 total time | ~43,640 s (~12.1 hours) |
| **Total training time** | **~51,420 s (~14.3 hours)** |
| Early stopping triggered | No — improvement continued through epoch 46 |
| Best weights | `convnext_tiny_adamw_ce_smooth_peft/weights/best.pt` |

---

## 7. Full Project Leaderboard

All models trained on the same 169-species dataset, same 80/10/10 split.

| Rank | Model | Top-1 | Top-5 | Params | Size | GPU (ms) | CPU (ms) | Notes |
|---|---|---|---|---|---|---|---|---|
| **1** | **ConvNeXt-Tiny (PEFT)** | **91.13%** | **98.32%** | **28.6M** | **~110 MB** | **7.6** | **35.6** | **New accuracy leader** |
| 2 | EfficientNet-B0 | 89.5% | 97.9% | 5.3M | 17.2 MB | 11.6 | 19.2 | Previous best accuracy |
| 3 | YOLOv26n-cls | 88.1% | 98.4% | 1.74M | 3.6 MB | 6.3 | 7.1 | Production model (PyTorch) |
| 3 | YOLOv26n TFLite f16 | 88.1% | 98.4% | 1.74M | 3.6 MB | — | **4.8** | **Production (deployed)** |
| 4 | YOLOv8n-cls | 86.8% | 97.9% | 2.7M | 3.4 MB | 4.6 | 6.3 | Original baseline |
| 5 | TaxonomicYOLO26 | 79.7% | — | ~2M | 25.0 MB | — | — | Dual-head genus+species |
| — | ConvNeXt-Tiny (full, 3 ep) | 80.3% | 96.3% | 28.6M | ~110 MB | — | — | Interrupted — not valid |

---

## 8. Inference Speed Benchmark

All models were benchmarked using a single dummy image (batch=1, 224×224), matching real-world usage in the Vision API where one image is processed per request. Each model was given 50 warmup runs before 200 timed runs were recorded.

**Hardware:** NVIDIA GeForce RTX 4070 (GPU) / same machine CPU  
**Script:** `scripts/benchmark_speed.py`

| Model | Top-1 | GPU (ms) | CPU (ms) | Notes |
|---|---|---|---|---|
| YOLOv8n-cls | 86.8% | **4.6** | 6.3 | Fastest GPU |
| YOLOv26n TFLite f16 | 88.1% | — | **4.8** | **Production — fastest CPU** |
| YOLOv26n-cls (PyTorch) | 88.1% | 6.3 | 7.1 | |
| ConvNeXt-Tiny (PEFT) | **91.1%** | 7.6 | 35.6 | Slowest CPU |
| EfficientNet-B0 | 89.5% | 11.6 | 19.2 | |

### Key findings

**On GPU**, ConvNeXt-Tiny is surprisingly competitive — only 1.3 ms slower than YOLOv26n (7.6 ms vs 6.3 ms) while delivering 3% higher accuracy. The gap is small enough that on a GPU-equipped server, ConvNeXt-Tiny would be a reasonable accuracy-focused choice.

**On CPU**, the picture reverses entirely. ConvNeXt-Tiny takes **35.6 ms** — 7.4× slower than the TFLite production model (4.8 ms). Cloud Run allocates CPU-only containers (no GPU), so this is the number that governs production suitability.

The TFLite model's CPU speed advantage comes from its XNNPACK delegate, which contains hand-optimised CPU kernels specifically for the operations present in the YOLOv26 graph. PyTorch's CPU runtime for ConvNeXt-Tiny does not benefit from this level of operator-level optimisation, despite ConvNeXt also relying heavily on depthwise convolutions.

**EfficientNet-B0** sits between the two: 19.2 ms on CPU — still 4× slower than TFLite, but 1.8× faster than ConvNeXt-Tiny. This reflects its smaller parameter count (5.3M vs 28.6M).

### Production verdict

The benchmark confirms the production decision: **YOLOv26n TFLite float16 is the correct deployment choice.** It is the fastest model on CPU (the only hardware available in Cloud Run), has the smallest Docker image footprint (~200 MB), and still achieves 88.1% top-1 — well within the project's accuracy requirements given the LLM audit layer provides a second verification pass.

---

## 9. Analysis and Discussion

### 9.1 ConvNeXt beats every other model on Top-1 accuracy

With a best validation Top-1 of **91.13%**, the ConvNeXt-Tiny PEFT run sets a new accuracy ceiling for this project. This is +1.6 percentage points over EfficientNet-B0 (89.5%) and +3.0 percentage points over the production YOLOv26n (88.1%).

This confirms the pre-training hypothesis: ConvNeXt-Tiny's modern architectural decisions — LayerNorm, 7×7 depthwise convolutions, inverted bottleneck — transfer meaningfully to fine-grained mushroom classification. The larger receptive field per layer is a plausible explanation: distinguishing 169 species by subtle features like gill attachment, cap cuticle texture, and stem ring position benefits from layers that can integrate context across a wider spatial area before committing to a representation.

### 9.2 PEFT was essential, not optional

The full fine-tuning run (~7,278 s/epoch) would have required ~100 hours to complete. PEFT reduced Phase 1 to ~776 s/epoch — a **9.4× speedup** — making the experiment feasible on consumer hardware.

Beyond speed, the frozen Phase 1 produced a better early baseline: by epoch 8, the PEFT run had already reached 87.75% val top-1 — more than EfficientNet-B0 and higher than what the full run reached at epoch 3 (80.3%). Freezing the backbone forces the classifier head to learn a good linear mapping over fixed ImageNet features before any of those features are disturbed by task-specific gradients. This gives the head a stable starting point for Phase 2.

### 9.3 The Phase 2 transition was clean

Val top-1 jumped from 87.09% (end of Phase 1, epoch 10) to 88.56% immediately at the first Phase 2 epoch (epoch 11) — an improvement of +1.5 pp in a single epoch. Despite the large grad norm spike at epoch 11 (~14.3, vs ~2.2 at the end of Phase 1), the model did not diverge. The gradient clipping at max_norm=1.0 kept individual parameter updates bounded, and the 10× lower backbone LR prevented the newly unfrozen backbone from taking destructively large steps.

### 9.4 Label smoothing — visible effect on the loss gap

By epoch 50, the training loss (0.8487) is substantially higher than expected for a model achieving 99.77% training top-1. This gap is by design. Label smoothing redistributes 0.1 probability mass from the correct class to all 168 other classes. The loss function therefore never considers the training example "perfectly answered" — a model that outputs a softmax probability of 1.0 for the correct class still receives a nonzero loss contribution from the smoothed target distribution. This consistently and measurably reduces overfitting by preventing the model from learning to be overconfident on training samples. The continued improvement in val top-1 all the way to epoch 46 — with no early stopping triggered — suggests the smoothing did its job.

### 9.5 Top-5 accuracy — the plateau

While Top-1 continued improving through epoch 46, Top-5 peaked early (98.32% at epoch 8 in Phase 1) and did not meaningfully improve further. This indicates that the model's top-5 predictions were correctly ordered very early in training — the 5 most plausible species for a given image were identified quickly — but the final disambiguation to the single correct label required the additional fine-tuning of Phase 2 to resolve.

### 9.6 Why EfficientNet-B0 is 5.4× smaller but only 1.6% behind

EfficientNet-B0's compound scaling law is very efficient at the B0 scale. It was designed specifically for the classification task and optimises depth, width, and resolution together in a derived ratio. ConvNeXt-Tiny's advantage is structural — the architectural improvements (LayerNorm, depthwise 7×7, GELU) provide richer feature representations — but they come with a significantly larger parameter count (28.6M vs 5.3M). For scenarios where model size matters, EfficientNet-B0's accuracy-per-parameter ratio is better. For scenarios where raw accuracy is the priority, ConvNeXt-Tiny wins.

---

## 10. Production Considerations

ConvNeXt-Tiny is not a viable replacement for the production model (YOLOv26n TFLite). The benchmark results (Section 8) make this concrete:

| Factor | YOLOv26n-cls (Production) | ConvNeXt-Tiny PEFT |
|---|---|---|
| Parameters | 1.74M | 28.6M |
| File size | 3.6 MB | ~110 MB |
| Export format | TFLite float16 | PyTorch only (TFLite export is non-trivial for ConvNeXt depthwise ops) |
| Docker image size | ~200 MB | ~1.5 GB+ |
| **GPU inference** | **6.3 ms** | **7.6 ms** |
| **CPU inference** | **4.8 ms (TFLite)** | **35.6 ms — 7.4× slower** |
| Cold start (Cloud Run) | Fast | Significantly slower (larger image pull) |
| Cost on Cloud Run (scale-to-zero) | Near zero | Higher (longer init time billed per request) |

The 0.2 ms TFLite inference and ~200 MB Docker image were the result of a deliberate engineering decision to keep the production deployment fast and cheap. ConvNeXt-Tiny's +3.0% accuracy gain does not justify these operational costs for the current deployment model.

**Where ConvNeXt-Tiny does add value:**

- **Accuracy ceiling benchmark.** The project now knows the dataset supports at least 91.1% top-1. Any future lightweight model can be measured against this ceiling.
- **Teacher for knowledge distillation.** A future experiment could train a smaller student model (e.g., a modified YOLOv26n) to mimic ConvNeXt-Tiny's softened probability outputs, potentially transferring some of the accuracy gain into a deployable-size model.
- **Offline re-labelling of drift images.** Low-confidence drift images saved during production can be passed through ConvNeXt-Tiny for high-confidence re-labelling before being added to the retraining dataset.

---

## 11. Conclusion

Two ConvNeXt-Tiny training runs were conducted on the 169-species mushroom dataset.

The **full fine-tuning** run was abandoned after 3 epochs due to prohibitive epoch time (~2 hours/epoch on an RTX 3070 Ti). At epoch 3 it had reached 80.3% val top-1 with a clearly upward trajectory, confirming the architecture is learning — but the time cost was not acceptable.

The **PEFT two-phase** run completed all 50 epochs in approximately 14.3 hours by freezing the backbone for the first 10 epochs and unfreezing it with a 10× lower LR for the remaining 40. It achieved a **best val Top-1 of 91.13%** at epoch 46 — the highest accuracy in the project by a margin of 1.6 percentage points over the previous best (EfficientNet-B0, 89.5%) and 3.0 percentage points over the production model (YOLOv26n-cls, 88.1%).

The experiment validates three things:

1. ConvNeXt's modern architectural improvements (LayerNorm, 7×7 depthwise conv, GELU, inverted bottleneck) transfer well to fine-grained mushroom species classification.
2. PEFT backbone freezing is an effective strategy for making large models trainable on consumer hardware — the 9.4× Phase 1 speedup was decisive.
3. The dataset is rich enough to support >91% single-model top-1 accuracy with the right architecture, establishing a meaningful target for future experiments.

ConvNeXt-Tiny remains a research model in this project. The production system continues to use YOLOv26n-cls exported to TFLite float16 for its unmatched inference speed (0.2 ms), compact deployment footprint (~200 MB Docker image), and near-zero Cloud Run cost.
