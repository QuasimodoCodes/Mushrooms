"""
Shared dataloader for DINOv2 experiments.

Uses identical augmentation to scripts/training/vit/dataset.py so that
results are directly comparable across all Herman-family models
(ViT-S/16, ConvNeXt, DINOv2).

DINOv2 was pretrained with ViT-S/14 patches on 224×224 ImageNet-normalised
images — same normalisation as our existing loaders, so no changes needed.
"""

import os
import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMG_SIZE   = 224
BATCH_SIZE = 32

_DEFAULT_WORKERS = 0 if sys.platform == "win32" else 4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_dataloaders(data_dir, batch_size=BATCH_SIZE, num_workers=_DEFAULT_WORKERS):
    """
    Return (train_loader, val_loader, test_loader, class_names).

    data_dir must contain train/ val/ test/ subfolders in ImageFolder format —
    the same layout produced by the YOLO dataset split script.

    Augmentation
    ------------
    Training (same as vit/dataset.py for fair comparison):
        RandomResizedCrop(224, scale=0.6-1.0)
        RandomHorizontalFlip
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.015)
        RandomRotation(15°)
        Normalize(ImageNet mean/std)

    Val / Test:
        Resize(256) → CenterCrop(224) → Normalize
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

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfm)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=eval_tfm)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=eval_tfm)

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

    return train_loader, val_loader, test_loader, train_ds.classes
