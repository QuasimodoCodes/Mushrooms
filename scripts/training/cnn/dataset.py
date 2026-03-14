import os
import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMG_SIZE    = 224
BATCH_SIZE  = 32

# Windows uses 'spawn' for multiprocessing — worker processes re-import the
# calling script from scratch, which breaks when the script isn't a clean module.
# Setting num_workers=0 disables multiprocessing and loads data on the main thread.
# On Linux/Mac (fork-based), 4 workers is safe and gives a real speedup.
_DEFAULT_WORKERS = 0 if sys.platform == "win32" else 4

# ImageNet mean/std — required when using ImageNet-pretrained weights.
# The backbone has already "learned" to expect pixels normalised this way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_dataloaders(data_dir, batch_size=BATCH_SIZE, num_workers=_DEFAULT_WORKERS):
    """
    Return (train_loader, val_loader, test_loader, class_names) for the
    mushroom dataset at data_dir/train|val|test.

    Augmentation strategy
    ---------------------
    Training:
        RandomResizedCrop   — forces the model to recognise mushrooms at varying
                              zoom levels and crop positions. Critical for real-world
                              photos where subjects aren't always centred.
        RandomHorizontalFlip — mushrooms are symmetric left/right; free extra data.
        ColorJitter          — simulates different lighting conditions, shadow,
                              camera white balance. Huge variance in the wild.
        RandomRotation       — mushrooms can appear at any angle in the field.
        ToTensor + Normalize — converts [0,255] uint8 → [0,1] float, then shifts
                              to ImageNet mean/std space.

    Validation / Test:
        Resize(256) → CenterCrop(224) — standard ImageNet eval protocol.
        No random augmentation — we need a stable, reproducible metric.
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

    # ImageFolder reads subfolders as class labels — identical to how YOLO reads
    # the dataset_split directory, so no data reorganisation is needed.
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfm)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),   transform=eval_tfm)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  transform=eval_tfm)

    # pin_memory=True speeds up CPU→GPU transfers but requires a CUDA device.
    # We disable it automatically when no GPU is present to avoid the warning.
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
