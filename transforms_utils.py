"""
Transforms module for the Plant Health Checker project.

Defines:
- Basic transform (resize + tensor only) used for initial dataset exploration
- Training transform with data augmentation
- Validation/Test transform with normalization only
"""

import torchvision.transforms as transforms

import config


def get_basic_transform():
    """
    Minimal transform used for initial dataset loading/exploration.
    Only resizes and converts to tensor (no normalization).
    """
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor()
    ])


def get_train_transform():
    """
    Training transform with data augmentation:
    random horizontal flip, rotation, color jitter, then ImageNet normalization.
    """
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])


def get_val_test_transform():
    """
    Validation/Test transform: resize + ImageNet normalization only,
    no augmentation.
    """
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
    ])


def denormalize(tensor):
    """
    Reverse ImageNet normalization for visualization purposes.

    Args:
        tensor: normalized image tensor of shape (C, H, W)

    Returns:
        Denormalized tensor clamped to [0, 1].
    """
    import torch
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    return torch.clamp(img, 0, 1)
