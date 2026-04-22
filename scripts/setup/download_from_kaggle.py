"""
download_from_kaggle.py — Download and prepare the mushroom dataset from Kaggle.

Usage (from project root):
    python scripts/setup/download_from_kaggle.py

Requirements:
    pip install kaggle split-folders

Kaggle credentials:
    1. Go to https://www.kaggle.com/settings/account
    2. Click "Create New Token" — downloads kaggle.json
    3. Place kaggle.json in:
         Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json
         Linux/Mac: ~/.kaggle/kaggle.json

What this script does:
    1. Downloads zlatan599/mushroom1 from Kaggle (~12 GB)
    2. Splits images into 80/10/10 train/val/test by species
    3. Saves to data/dataset_split/ — ready for YOLO and PyTorch training
"""

import os
import sys
import shutil
import zipfile

# ─── Paths ────────────────────────────────────────────────────────────────────

_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.abspath(os.path.join(_HERE, "..", ".."))
RAW_DIR   = os.path.join(_ROOT, "data", "kaggle_raw")
SPLIT_DIR = os.path.join(_ROOT, "data", "dataset_split")

KAGGLE_DATASET = "zlatan599/mushroom1"


# ─── Step 1: Download from Kaggle ─────────────────────────────────────────────

def download():
    try:
        import kaggle
    except ImportError:
        print("ERROR: kaggle package not installed.")
        print("       Run:  pip install kaggle")
        sys.exit(1)

    print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
    print(f"Saving to: {RAW_DIR}\n")
    os.makedirs(RAW_DIR, exist_ok=True)

    os.system(f'kaggle datasets download -d {KAGGLE_DATASET} -p "{RAW_DIR}" --unzip')
    print("\nDownload complete.")


# ─── Step 2: Find the images root ─────────────────────────────────────────────

def find_images_root(base):
    """
    The Kaggle dataset may unzip into a nested folder.
    Walk until we find a directory that contains species subfolders.
    Returns the path to the folder that has species as direct children.
    """
    for root, dirs, files in os.walk(base):
        # A species folder contains image files directly
        if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
            return os.path.dirname(root)
    return base


# ─── Step 3: Split into train/val/test ────────────────────────────────────────

def split():
    try:
        import splitfolders
    except ImportError:
        print("ERROR: split-folders package not installed.")
        print("       Run:  pip install split-folders")
        sys.exit(1)

    images_root = find_images_root(RAW_DIR)
    print(f"\nFound images root: {images_root}")
    print(f"Splitting 80/10/10 → {SPLIT_DIR}\n")

    if os.path.exists(SPLIT_DIR):
        print(f"WARNING: {SPLIT_DIR} already exists — removing it first.")
        shutil.rmtree(SPLIT_DIR)

    splitfolders.ratio(
        images_root,
        output=SPLIT_DIR,
        seed=42,
        ratio=(0.8, 0.1, 0.1),
        group_prefix=None,
    )
    print("\nSplit complete.")


# ─── Step 4: Verify ───────────────────────────────────────────────────────────

def verify():
    print("\nVerifying dataset_split...")
    for split in ("train", "val", "test"):
        split_path = os.path.join(SPLIT_DIR, split)
        if not os.path.exists(split_path):
            print(f"  ERROR: {split}/ folder missing!")
            continue
        n_classes = len(os.listdir(split_path))
        n_images  = sum(len(files) for _, _, files in os.walk(split_path))
        print(f"  {split:>5}/  {n_classes} species  {n_images:>7,} images")

    print("\nDataset ready. You can now run:")
    print("  python scripts/training/convnext/train.py")
    print("  python scripts/training/vit/train.py")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    download()
    split()
    verify()
