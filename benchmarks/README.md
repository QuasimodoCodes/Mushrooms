# Benchmarks

This folder contains experiments that validate our model architecture against published research on standard public datasets. These benchmarks are separate from our production 169-species forager system.

---

## Why Benchmark?

Our production model was trained on a custom 169-species Kaggle dataset. To situate our results within the broader research community and support publication, we fine-tune the same YOLOv26n architecture on public benchmark datasets and compare against published baselines.

---

## DF24-Mini

**Dataset:** Danish Fungi 2024 – Mini (Picek et al., WACV 2022 — updated split July 2024)
**Classes:** 182 species (6 genera: Amanita, Boletus, Russula, Agaricus, Mycena, Clitocybe)
**Images:** ~36,000 (12.5GB, same images as DF20-Mini)
**Split:** Train / Public Test (ObservationID-based — no leakage between splits)

### Dataset Overlap with Our Production Model

Our production model was trained on 169 species from a Kaggle forager-focused dataset. DF24-Mini covers 182 species drawn from Danish citizen science observations. When we compared the two species lists, only **11 species appear in both datasets** — roughly 6% overlap. This means these are effectively two independent datasets with almost entirely different species, which makes the benchmark a genuine test of how well the YOLOv26n architecture generalises rather than just a re-test of familiar classes.

The 11 shared species are all common European genera: *Amanita muscaria*, *Amanita phalloides*, *Amanita pantherina*, *Amanita citrina*, *Amanita rubescens*, *Boletus edulis*, *Boletus reticulatus*, *Clitocybe nebularis*, *Mycena haematopus*, *Agaricus augustus*, *Agaricus xanthodermus*.

### Process

1. Download DF24-Mini images and metadata from `ptak.felk.cvut.cz`
2. Convert metadata CSVs into a folder structure compatible with Ultralytics YOLO classification
3. Fine-tune `yolo26n-cls.pt` on DF24-Mini training split (182 classes, imgsz=224)
4. Evaluate on DF24-Mini public test split
5. Compare Top-1 / Top-3 against published baselines at 224×224

### Published Baselines (DF24-Mini, 224×224)

| Model | Top-1 | Top-3 | F1 |
|:------|:------|:------|:---|
| ViT-Large/16 | 67.52% | 84.46% | 55.90% |
| ViT-Base/16 | 65.33% | 82.44% | 52.28% |
| SE-ResNeXt-101 | 62.42% | 80.71% | 50.01% |
| EfficientNet-B0 | 58.58% | 77.01% | 46.00% |
| EfficientNet-B3 | 59.31% | 78.79% | 47.83% |

Source: Picek et al. — [Danish Fungi 2020 (WACV 2022)](https://openaccess.thecvf.com/content/WACV2022/html/Picek_Danish_Fungi_2020_-_Not_Just_Another_Image_Recognition_Dataset_WACV_2022_paper.html)

### Our Results

| Model | Top-1 | Top-3 | F1 | Notes |
|:------|:------|:------|:---|:------|
| YOLOv26n-cls | TBD | TBD | TBD | imgsz=224, fine-tuned from yolo26n-cls.pt |

*Results will be updated once training completes.*

---

## Citation

```
@inproceedings{picek2022danish,
  title={Danish fungi 2020-not just another image recognition dataset},
  author={Picek, Luk{\'a}{\v{s}} and {\v{S}}ulc, Milan and Matas, Ji{\v{r}}{\'\i} and Jeppesen, Thomas S and Heilmann-Clausen, Jacob and L{\ae}ss{\o}e, Thomas and Fr{\o}slev, Tobias},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={1525--1535},
  year={2022}
}
```
