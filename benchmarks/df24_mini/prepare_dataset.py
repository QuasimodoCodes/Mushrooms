"""
Converts DF24-Mini CSV metadata into a YOLO-compatible classification folder structure.

Input (from downloaded dataset):
    data/images/DF20/<ImageUniqueID>.JPG
    data/metadata/DanishFungi2024-Mini-train.csv
    data/metadata/DanishFungi2024-Mini-pubtest.csv

Output (YOLO classification format):
    data/yolo_ready/
        train/
            Russula_fragilis/
                image1.jpg ...
        val/
            Russula_fragilis/
                image1.jpg ...

Uses hardlinks to avoid duplicating 12.5GB of images.
Falls back to file copy if hardlinks fail.
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
IMAGES_DIR  = BASE / "data" / "images" / "DF20M-images" / "DF20M"
META_TRAIN  = BASE / "data" / "metadata" / "DanishFungi2024-Mini-train.csv"
META_TEST   = BASE / "data" / "metadata" / "DanishFungi2024-Mini-pubtest.csv"
OUT_DIR     = BASE / "data" / "yolo_ready"


def species_to_folder(name: str) -> str:
    return name.strip().replace(" ", "_")


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_split(csv_path: Path, split_name: str):
    df = pd.read_csv(csv_path)
    out_split = OUT_DIR / split_name
    missing = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
        if pd.isna(row["species"]):
            missing += 1
            continue

        # image_path in CSV: "DF20/2238472345-167057.JPG"
        img_filename = Path(row["image_path"]).name
        src = IMAGES_DIR / img_filename

        if not src.exists():
            missing += 1
            continue

        folder = species_to_folder(row["species"])
        dst = out_split / folder / img_filename
        link_or_copy(src, dst)

    print(f"[{split_name}] Done. Missing images: {missing}/{len(df)}")


if __name__ == "__main__":
    print("Checking image directory...")
    if not IMAGES_DIR.exists():
        print(f"ERROR: Images not found at {IMAGES_DIR}")
        print("Make sure you extracted DF20M-images.tar.gz first.")
        exit(1)

    print(f"Found images at: {IMAGES_DIR}")
    print(f"Output will be written to: {OUT_DIR}\n")

    build_split(META_TRAIN, "train")
    build_split(META_TEST,  "val")

    # Count classes and images
    train_classes = list((OUT_DIR / "train").iterdir())
    print(f"\nDataset ready:")
    print(f"  Train classes : {len(train_classes)}")
    print(f"  Output path   : {OUT_DIR}")
