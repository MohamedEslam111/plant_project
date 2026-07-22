"""
Dataset module for the Plant Health Checker project.

Handles:
- Loading the PlantVillage dataset
- Extracting plant species information from class names
- Splitting the dataset into train/val/test subsets
- Computing class distribution statistics
"""

from collections import Counter

import torch
from torch.utils.data import random_split, Subset
from torchvision.datasets import ImageFolder

import config


def load_raw_dataset(transform):
    """
    Load the PlantVillage dataset with a given transform applied.

    Args:
        transform: torchvision transform to apply to each image.

    Returns:
        ImageFolder dataset instance.
    """
    dataset = ImageFolder(root=config.DATA_DIR, transform=transform)
    return dataset


def print_dataset_overview(dataset) -> None:
    """Print basic information about the loaded dataset."""
    class_names = dataset.classes
    num_classes = len(class_names)
    total_images = len(dataset)

    print(f"Total number of images: {total_images}")
    print(f"Number of classes: {num_classes}")
    print("\nFirst 10 classes:")
    for i, class_name in enumerate(class_names[:10]):
        print(f"  {i}: {class_name}")


def extract_species_info(class_names):
    """
    Group class names by plant species (e.g., Apple, Tomato, Potato).

    Args:
        class_names: list of class name strings from ImageFolder.classes

    Returns:
        dict mapping species name -> list of (class_index, class_name) tuples
    """
    species_dict = {}

    for idx, class_name in enumerate(class_names):
        if class_name.startswith('Tomato'):
            species = 'Tomato'
        elif class_name.startswith('Potato'):
            species = 'Potato'
        elif class_name.startswith('Pepper'):
            species = 'Pepper__bell'
        else:
            species = class_name.split('_')[0]

        if species not in species_dict:
            species_dict[species] = []
        species_dict[species].append((idx, class_name))

    return species_dict


def print_species_overview(species_dict) -> None:
    """Print the number of disease classes per plant species."""
    print(f"Plant Species in Dataset ({len(species_dict)} species):\n")
    for species, classes in species_dict.items():
        print(f"{species}: {len(classes)} disease classes")


def split_dataset(full_dataset):
    """
    Split the full dataset into train/val/test subsets using the ratios
    defined in config.py.

    Args:
        full_dataset: the complete ImageFolder dataset.

    Returns:
        (train_dataset, val_dataset, test_dataset) as Subset objects.
    """
    total_images = len(full_dataset)

    train_size = int(config.TRAIN_SPLIT * total_images)
    val_size = int(config.VAL_SPLIT * total_images)
    test_size = total_images - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )

    print(f"Training samples: {len(train_dataset)} ({len(train_dataset) / total_images * 100:.1f}%)")
    print(f"Validation samples: {len(val_dataset)} ({len(val_dataset) / total_images * 100:.1f}%)")
    print(f"Testing samples: {len(test_dataset)} ({len(test_dataset) / total_images * 100:.1f}%)")

    return train_dataset, val_dataset, test_dataset


def get_class_distribution(dataset, dataset_name: str, class_names):
    """
    Compute and print the class distribution for a dataset split.

    Args:
        dataset: a Subset of the full dataset (must expose .dataset and .indices)
        dataset_name: label used for printing (e.g. "Training")
        class_names: list of all class name strings.

    Returns:
        Counter mapping class_index -> image count.
    """
    labels = [dataset.dataset.targets[idx] for idx in dataset.indices]
    class_counts = Counter(labels)

    print(f"\n{dataset_name} Set - Top 10 Classes:")
    for class_idx, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {class_names[class_idx]}: {count} images")

    return class_counts


def rebuild_subsets_with_transforms(train_dataset, val_dataset, test_dataset,
                                     train_transform, val_test_transform):
    """
    Rebuild the train/val/test subsets using proper transforms
    (augmentation for training, plain normalization for val/test),
    while preserving the exact same split indices.

    Args:
        train_dataset, val_dataset, test_dataset: original Subset splits
            (used only to extract their .indices).
        train_transform: transform with augmentation, used for training data.
        val_test_transform: transform without augmentation, used for val/test data.

    Returns:
        (train_dataset_processed, val_dataset_processed, test_dataset_processed)
    """
    full_dataset_train = ImageFolder(root=config.DATA_DIR, transform=train_transform)
    full_dataset_val = ImageFolder(root=config.DATA_DIR, transform=val_test_transform)
    full_dataset_test = ImageFolder(root=config.DATA_DIR, transform=val_test_transform)

    train_dataset_processed = Subset(full_dataset_train, train_dataset.indices)
    val_dataset_processed = Subset(full_dataset_val, val_dataset.indices)
    test_dataset_processed = Subset(full_dataset_test, test_dataset.indices)

    print("Preprocessed datasets created:")
    print(f"  Training: {len(train_dataset_processed)} images (with augmentation)")
    print(f"  Validation: {len(val_dataset_processed)} images (no augmentation)")
    print(f"  Testing: {len(test_dataset_processed)} images (no augmentation)")

    return train_dataset_processed, val_dataset_processed, test_dataset_processed
