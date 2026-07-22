"""
Visualization module for the Plant Health Checker project.

Handles:
- Visualizing random samples from a dataset split
- Visualizing one sample per plant species
- Comparing original vs preprocessed (augmented + normalized) images
- Visualizing a batch of preprocessed images from a DataLoader
"""

import numpy as np
import matplotlib.pyplot as plt

from transforms_utils import denormalize


def visualize_samples(dataset, dataset_name, class_names, num_samples=12, figsize=(15, 10)):
    """Visualize random samples from a dataset split."""
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(3, 4, figsize=figsize)
    axes = axes.ravel()

    for idx, ax in zip(indices, axes):
        image, label = dataset[idx]
        img_np = image.permute(1, 2, 0).numpy()

        ax.imshow(img_np)
        ax.set_title(class_names[label].replace('___', '\n'), fontsize=8)
        ax.axis('off')

    plt.suptitle(f'{dataset_name} Set - Random Samples', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.show()


def visualize_species_samples(dataset, species_dict, class_names):
    """Visualize one sample image from each plant species."""
    species_list = list(species_dict.keys())
    num_species = len(species_list)

    fig, axes = plt.subplots(1, num_species, figsize=(5 * num_species, 5))

    if num_species == 1:
        axes = [axes]

    for idx, species in enumerate(species_list):
        class_idx = species_dict[species][0][0]

        # Find an image belonging to this class within the dataset
        for i in range(len(dataset)):
            image, label = dataset[i]
            if label == class_idx:
                img_np = image.permute(1, 2, 0).numpy()
                axes[idx].imshow(img_np)

                # Build a clean disease name from the full class name
                full_class_name = class_names[label]
                disease_name = full_class_name.replace(species, "").strip('_')
                disease_name = disease_name.replace('___', ' ').replace('__', ' ').replace('_', ' ')

                axes[idx].set_title(f"{species}\n{disease_name}", fontsize=10)
                axes[idx].axis('off')
                break

    plt.suptitle('Samples from Different Plant Species', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.show()


def compare_preprocessing(train_dataset, train_dataset_processed, class_names):
    """Compare an original (unprocessed) image against its preprocessed version."""
    idx = np.random.randint(len(train_dataset))

    # Original (no normalization)
    original_img, label = train_dataset[idx]

    # Preprocessed (with augmentation and normalization)
    processed_img, _ = train_dataset_processed[idx]
    processed_img = denormalize(processed_img)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(original_img.permute(1, 2, 0).numpy())
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(processed_img.permute(1, 2, 0).numpy())
    axes[1].set_title('Preprocessed (Augmented + Normalized)', fontsize=12)
    axes[1].axis('off')

    plt.suptitle(f'Class: {class_names[label]}', fontsize=14)
    plt.tight_layout()
    plt.show()


def visualize_preprocessed_batch(dataloader, title="Preprocessed Images"):
    """Visualize a batch of preprocessed images from a DataLoader."""
    images, labels = next(iter(dataloader))
    num_to_show = min(8, images.size(0))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for idx in range(num_to_show):
        img = denormalize(images[idx])
        img_np = img.permute(1, 2, 0).numpy()
        axes[idx].imshow(img_np)
        axes[idx].axis('off')

    # Hide any unused subplots if the batch had fewer than 8 images
    for idx in range(num_to_show, 8):
        axes[idx].axis('off')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def test_dataloader_iteration(train_loader, val_loader, test_loader):
    """Print shape and value statistics for one batch from each DataLoader."""
    print("Testing DataLoader iteration...\n")

    for split_name, loader in [("Training", train_loader), ("Validation", val_loader), ("Testing", test_loader)]:
        images, labels = next(iter(loader))
        print(f"{split_name} Set:")
        print(f"  Batch shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Min pixel value: {images.min():.3f}")
        print(f"  Max pixel value: {images.max():.3f}")
        print(f"  Mean: {images.mean():.3f}")
        print(f"  Std: {images.std():.3f}")
        print()
