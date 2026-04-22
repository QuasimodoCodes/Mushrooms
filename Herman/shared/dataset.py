"""
Shared dataloader for Herman experiments (ViT + ConvNeXt).

Identical augmentation strategy to scripts/training/cnn/dataset.py so that
results are directly comparable. Both ViT-S/16 and ConvNeXt-Tiny use 224×224
ImageNet-normalised inputs, so no changes are needed.
"""

import os
import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMG_SIZE   = 224
BATCH_SIZE = 32

# Windows uses 'spawn' — multiprocessing workers re-import from scratch, which
# breaks non-module scripts. num_workers=0 disables it safely on Windows.
_DEFAULT_WORKERS = 0 if sys.platform == "win32" else 4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_dataloaders(data_dir, batch_size=BATCH_SIZE, num_workers=_DEFAULT_WORKERS):
    """
    Return (train_loader, val_loader, test_loader, class_names).

    data_dir must contain train/ val/ test/ subfolders, each with one
    subfolder per species (ImageFolder format — same layout as the YOLO split).

    Augmentation
    ------------
    Training:
        RandomResizedCrop   — scale (0.6–1.0), forcing the model to handle
                              varied zoom and off-centre framing.
        RandomHorizontalFlip — free augmentation (mushrooms are left-right symmetric).
        ColorJitter          — lighting variance common in field photography.
        RandomRotation(15°)  — mushrooms can appear at any angle.
        Normalize            — ImageNet mean/std required for pretrained backbones.

    Val / Test:
        Resize(256) → CenterCrop(224) — standard ImageNet eval protocol.
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
