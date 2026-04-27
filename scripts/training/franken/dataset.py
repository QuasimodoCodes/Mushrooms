"""
Taxonomic dataset for the Multi-Branch Taxonomic Classifier.

Wraps torchvision ImageFolder but returns THREE items per sample:
    (image, genus_label, species_label)

The genus is derived at runtime by splitting the folder name on the first
space — e.g. "Amanita muscaria" → genus "Amanita".  No data reorganisation
or separate label files are needed; the split is purely in memory.

num_genera is calculated dynamically from whatever classes exist in the
dataset root, so adding new species never requires touching this file.
"""

import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

IMG_SIZE   = 224
BATCH_SIZE = 32

# Windows multiprocessing constraint — same reasoning as cnn/dataset.py
_DEFAULT_WORKERS = 0 if sys.platform == "win32" else 4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class TaxonomicDataset(Dataset):
    """
    Thin wrapper around ImageFolder that adds a genus label.

    Parameters
    ----------
    root : str
        Path to a split folder (train / val / test) whose subfolders are
        named with binomial nomenclature, e.g. "Amanita muscaria".
    transform : callable, optional
        Torchvision transform applied to each image.

    Attributes
    ----------
    classes     : list[str]   — all species names, sorted (from ImageFolder)
    genera      : list[str]   — unique genus names, sorted
    num_species : int
    num_genera  : int
    """

    def __init__(self, root: str, transform=None):
        self._base = datasets.ImageFolder(root, transform=transform)

        species_names = self._base.classes          # e.g. ["Amanita muscaria", …]
        genera        = sorted(set(n.split()[0] for n in species_names))

        self.classes     = species_names
        self.genera      = genera
        self.num_species = len(species_names)
        self.num_genera  = len(genera)

        self._genus_to_idx = {g: i for i, g in enumerate(genera)}

        # For every species index, pre-compute its genus index once.
        self._species_to_genus = {
            sp_idx: self._genus_to_idx[name.split()[0]]
            for sp_idx, name in enumerate(species_names)
        }

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx):
        img, species_label = self._base[idx]
        genus_label = self._species_to_genus[species_label]
        return img, genus_label, species_label


def get_dataloaders(data_dir: str, batch_size: int = BATCH_SIZE,
                    num_workers: int = _DEFAULT_WORKERS):
    """
    Return (train_loader, val_loader, test_loader, dataset_meta) where
    dataset_meta is a dict with keys: class_names, genera, num_species, num_genera.

    Augmentation strategy mirrors cnn/dataset.py exactly — RandomResizedCrop,
    HorizontalFlip, ColorJitter, RandomRotation for training; Resize+CenterCrop
    for evaluation.
    """
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.015),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    eval_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = TaxonomicDataset(os.path.join(data_dir, "train"), transform=train_tfm)
    val_ds   = TaxonomicDataset(os.path.join(data_dir, "val"),   transform=eval_tfm)
    test_ds  = TaxonomicDataset(os.path.join(data_dir, "test"),  transform=eval_tfm)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    meta = {
        "class_names": train_ds.classes,
        "genera":      train_ds.genera,
        "num_species": train_ds.num_species,
        "num_genera":  train_ds.num_genera,
    }

    return train_loader, val_loader, test_loader, meta
