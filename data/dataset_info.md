# Dataset Information

## Source
Dataset downloaded from Kaggle: `zlatan599/mushroom1`

## Structure
The original dataset contained images for 169 different mushroom species.
We have restructured this into an 80/10/10 (Train/Val/Test) split suitable for YOLO classification.

*   `dataset_split/train/`: 80% of images per species
*   `dataset_split/val/`: 10% of images per species
*   `dataset_split/test/`: 10% of images per species
*   `dataset.yaml`: YOLO configuration file pointing to the splits and defining the 169 class names.
*   `mushroom_species.json`: A master list of all 169 species names.

## Additional Metadata (CSV Files)
The downloaded dataset also included three CSV files:
*   `train.csv` (contains ~689k lines, mapping image paths to their target label)
*   `val.csv`
*   `test.csv`

**Current Usage**: We are currently using the `split-folders` approach because YOLO reads class names directly from folder structures (e.g., `train/Amanita_muscaria/image.jpg`). The CSV files are redundant for basic YOLO image classification, but they are preserved here in case they contain additional metadata or we want to switch to a custom PyTorch DataLoader later.
