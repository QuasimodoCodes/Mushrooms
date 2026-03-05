import kagglehub
import os
import json
import splitfolders
import shutil

# Step 1: Download the dataset using kagglehub
print("Downloading 'zlatan599/mushroom1' dataset...")
path = kagglehub.dataset_download("zlatan599/mushroom1")
print("Dataset downloaded to:", path)

# Step 2: Read the folder names (species/classes)
print("Extracting species names...")
species_list = []
if os.path.exists(path):
    # Depending on how the dataset is structured, we might need to find the directory containing the images.
    # Usually it's either in the root `path` or inside a subfolder. Let's list contents.
    contents = os.listdir(path)
    
    # Let's assume the root path contains the class folders directly, or there's one folder inside holding them.
    # We will print the contents to debug if needed.
    print(f"Contents of {path}:", contents)
    
    # We will try to find the directory with the class folders
    data_dir = path
    # If there's only one directory inside, maybe it's the root of the class folders
    if len(contents) == 1 and os.path.isdir(os.path.join(path, contents[0])):
        data_dir = os.path.join(path, contents[0])
    elif "train" in contents or "val" in contents:
        # If it's already split, we can just read train folders
        if "train" in contents:
            data_dir = os.path.join(path, "train")

    species_list = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"Found {len(species_list)} species.")

# Step 3: Save the folder names into a text file
output_classes_file = "mushroom_species.json"
with open(output_classes_file, "w") as f:
    json.dump(species_list, f, indent=4)
print(f"Saved species list to {output_classes_file}")

# Step 4: Split the images into training, validation, and testing sets (80/10/10)
# splitfolders will copy and split the images into a new directory structure
output_split_dir = "dataset_split"
if os.path.exists(output_split_dir):
    print("Removing existing split directory...")
    shutil.rmtree(output_split_dir)

print(f"Splitting dataset into train, val, and test (80/10/10) into '{output_split_dir}'...")
splitfolders.ratio(data_dir, output=output_split_dir,
    seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False) # set move=False to copy
print("Dataset split complete.")
