import os
import json
import splitfolders
import shutil

# Correct data directory
data_dir = r"C:\Users\spide\.cache\kagglehub\datasets\zlatan599\mushroom1\versions\2\merged_dataset"

# Extract species names
species_list = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
species_list.sort() # Sort to ensure consistent class numbering
print(f"Found {len(species_list)} species.")

# Save the folder names into a text file
output_classes_file = "mushroom_species.json"
with open(output_classes_file, "w") as f:
    json.dump(species_list, f, indent=4)
print(f"Saved species list to {output_classes_file}")

# Split the images
output_split_dir = "dataset_split"
if os.path.exists(output_split_dir):
    print("Removing existing split directory...")
    shutil.rmtree(output_split_dir)

print(f"Splitting dataset into train, val, and test (80/10/10) into '{output_split_dir}'...")
splitfolders.ratio(data_dir, output=output_split_dir,
    seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False)
print("Dataset split complete.")

# Create dataset.yaml for YOLO
yaml_content = f"""path: {os.path.abspath(output_split_dir)}
train: train
val: val
test: test

nc: {len(species_list)}
names:
"""
for idx, name in enumerate(species_list):
    yaml_content += f"  {idx}: {name}\n"

with open("dataset.yaml", "w") as f:
    f.write(yaml_content)
    
print("Created dataset.yaml")
