import os
from huggingface_hub import snapshot_download

# Step 1: Download the dataset using huggingface_hub
print("Downloading 'QuasimodoDodo/Mushrooms' dataset from Hugging Face...")

# We download the dataset directly to the 'data' directory.
# The `snapshot_download` automatically handles caching and downloading
# only what is needed.
try:
    path = snapshot_download(
        repo_id="QuasimodoDodo/Mushrooms",
        repo_type="dataset",
        local_dir="data",
        local_dir_use_symlinks=False # Set to False so we get actual files in the data dir
    )
    print("Dataset successfully downloaded to:", path)
    print("\nThe dataset is perfectly structured for YOLO in 'data/dataset_split/'.")
except Exception as e:
    print(f"Error downloading dataset: {e}")
    print("Please make sure you have access to the repository on Hugging Face.")
    exit(1)
