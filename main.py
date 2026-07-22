"""
Plant Health Checker - Main Pipeline
======================================
Loading, visualizing, and preprocessing the PlantVillage dataset, then
training and evaluating multiple transfer-learning models (ResNet50,
DenseNet121, MobileNetV3-Large, and ConvNeXt-Tiny) for side-by-side
comparison.

This script runs the full pipeline in order:
  1. Load dataset and explore
  2. Extract plant species information
  3. Split dataset into Train/Val/Test
  4. Visualize class distribution across splits
  5. Visualize sample images (random + per-species)
  6. Define preprocessing/augmentation transforms
  7. Apply preprocessing to datasets
  8. Create DataLoaders
  9. Visualize preprocessed images & compare with originals
  10. Test DataLoader iteration
  11. For each model in config.MODELS_TO_COMPARE:
        - Build the model
        - Train it (with early stopping + best-checkpoint saving)
        - Evaluate it on the test set
  12. Compare all trained models side-by-side
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import config
import dataset
import transforms_utils
import visualization
import model as model_module
import train as train_module
import evaluate
import compare_models


def prepare_data():
    """Run all data loading, splitting, preprocessing, and DataLoader steps."""
    # ------------------------------------------------------------
    # 1. Load dataset and explore
    # ------------------------------------------------------------
    basic_transform = transforms_utils.get_basic_transform()
    full_dataset = dataset.load_raw_dataset(basic_transform)

    class_names = full_dataset.classes
    num_classes = len(class_names)
    total_images = len(full_dataset)

    dataset.print_dataset_overview(full_dataset)

    # ------------------------------------------------------------
    # 2. Extract plant species information
    # ------------------------------------------------------------
    species_dict = dataset.extract_species_info(class_names)
    dataset.print_species_overview(species_dict)

    # ------------------------------------------------------------
    # 3. Split dataset into Train/Val/Test
    # ------------------------------------------------------------
    train_dataset, val_dataset, test_dataset = dataset.split_dataset(full_dataset)

    # ------------------------------------------------------------
    # 4. Visualize class distribution across splits
    # ------------------------------------------------------------
    dataset.get_class_distribution(train_dataset, "Training", class_names)
    dataset.get_class_distribution(val_dataset, "Validation", class_names)
    dataset.get_class_distribution(test_dataset, "Testing", class_names)

    # ------------------------------------------------------------
    # 5. Visualize sample images (random + per-species)
    # ------------------------------------------------------------
    visualization.visualize_samples(train_dataset, "Training", class_names)
    visualization.visualize_samples(val_dataset, "Validation", class_names)
    visualization.visualize_samples(test_dataset, "Testing", class_names)
    visualization.visualize_species_samples(train_dataset, species_dict, class_names)

    # ------------------------------------------------------------
    # 6. Define preprocessing/augmentation transforms
    # ------------------------------------------------------------
    train_transform = transforms_utils.get_train_transform()
    val_test_transform = transforms_utils.get_val_test_transform()

    print("Transformations defined:")
    print("\nTraining (with augmentation):")
    print(train_transform)
    print("\nValidation/Test (no augmentation):")
    print(val_test_transform)

    # ------------------------------------------------------------
    # 7. Apply preprocessing to datasets
    # ------------------------------------------------------------
    train_dataset_processed, val_dataset_processed, test_dataset_processed = \
        dataset.rebuild_subsets_with_transforms(
            train_dataset, val_dataset, test_dataset,
            train_transform, val_test_transform
        )

    # ------------------------------------------------------------
    # 8. Create DataLoaders
    # ------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset_processed, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset_processed, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset_processed, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True
    )

    print("DataLoaders created:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Testing batches: {len(test_loader)}")
    print(f"  Batch size: {config.BATCH_SIZE}")

    # ------------------------------------------------------------
    # 9. Visualize preprocessed images & compare with originals
    # ------------------------------------------------------------
    visualization.visualize_preprocessed_batch(train_loader, title="Preprocessed Training Images")

    for _ in range(3):
        visualization.compare_preprocessing(train_dataset, train_dataset_processed, class_names)

    # ------------------------------------------------------------
    # Dataset preparation summary
    # ------------------------------------------------------------
    print("=" * 60)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 60)
    print(f"\nDataset: PlantVillage Plant Disease Dataset")
    print(f"Total Images: {total_images:,}")
    print(f"Number of Classes: {num_classes}")
    print(f"Number of Plant Species: {len(species_dict)}")
    print(f"\nSpecies: {', '.join(list(species_dict.keys()))}")
    print(f"\nSplit Ratios:")
    print(f"  Training:   {len(train_dataset_processed):,} images "
          f"({len(train_dataset_processed) / total_images * 100:.1f}%)")
    print(f"  Validation: {len(val_dataset_processed):,} images "
          f"({len(val_dataset_processed) / total_images * 100:.1f}%)")
    print(f"  Testing:    {len(test_dataset_processed):,} images "
          f"({len(test_dataset_processed) / total_images * 100:.1f}%)")
    print(f"\nImage Specifications:")
    print(f"  Input Size: {config.IMG_SIZE}x{config.IMG_SIZE}")
    print(f"  Color Channels: 3 (RGB)")
    print(f"  Batch Size: {config.BATCH_SIZE}")
    print(f"\nPreprocessing:")
    print(f"  Normalization: ImageNet statistics "
          f"(mean={config.IMAGENET_MEAN}, std={config.IMAGENET_STD})")
    print(f"  Training Augmentation: Random flip, rotation, color jitter")
    print(f"  Val/Test Augmentation: None")
    print(f"\nReady for transfer learning!")
    print("=" * 60)

    # ------------------------------------------------------------
    # 10. Test DataLoader iteration
    # ------------------------------------------------------------
    visualization.test_dataloader_iteration(train_loader, val_loader, test_loader)

    data = {
        'class_names': class_names,
        'num_classes': num_classes,
        'total_images': total_images,
        'species_dict': species_dict,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
    }
    return data


def run_model_pipeline(model_name: str, data: dict):
    """
    Build, train, and evaluate a single model end-to-end.

    Args:
        model_name: key from model.MODEL_REGISTRY
            (e.g. 'resnet50', 'densenet121', 'mobilenet_v3_large', 'convnext_tiny').
        data: dict returned by prepare_data().

    Returns:
        result: dict with 'metrics', 'checkpoint', 'history',
                 'num_params', 'avg_inference_time_ms'.
    """
    class_names = data['class_names']
    num_classes = data['num_classes']
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    test_loader = data['test_loader']

    # ------------------------------------------------------------
    # 11a. Build model
    # ------------------------------------------------------------
    model = model_module.build_model(model_name, num_classes)
    num_params = evaluate.count_parameters(model)
    print(f"\n{model_name} - Trainable parameters: {num_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # Sanity check: one forward pass
    images, labels = next(iter(train_loader))
    images = images.to(config.DEVICE)
    outputs = model(images)
    print(f"Sanity check output shape: {outputs.shape}")

    # ------------------------------------------------------------
    # 11b. Train model
    # ------------------------------------------------------------
    history, checkpoint = train_module.train_model(model, model_name, train_loader, val_loader, optimizer)

    # ------------------------------------------------------------
    # 11c. Evaluate model on test set
    # ------------------------------------------------------------
    evaluate.plot_training_curves(history, model_name)

    all_preds, all_labels, avg_inference_time_ms = evaluate.run_inference_on_test_set(model, test_loader)
    metrics = evaluate.compute_overall_metrics(all_labels, all_preds)
    evaluate.save_classification_report(all_labels, all_preds, class_names, metrics, model_name)

    cm = evaluate.plot_confusion_matrix(all_labels, all_preds, class_names, model_name)
    evaluate.print_top_confusions(cm, class_names, top_n=10)
    evaluate.print_best_worst_classes(all_labels, all_preds, class_names, top_n=5)

    evaluate.visualize_prediction_samples(model, test_loader, class_names, model_name)

    evaluate.print_final_summary(
        data['total_images'], num_classes, data['species_dict'],
        data['train_dataset'], data['val_dataset'], data['test_dataset'],
        checkpoint, metrics, model_name
    )

    result = {
        'metrics': metrics,
        'checkpoint': checkpoint,
        'history': history,
        'num_params': num_params,
        'avg_inference_time_ms': avg_inference_time_ms,
    }
    return result


def main():
    # ------------------------------------------------------------
    # 0. Setup
    # ------------------------------------------------------------
    config.set_seed()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {config.DEVICE}")
    print(f"Dataset directory: {config.DATA_DIR}")
    print(f"Image size: {config.IMG_SIZE}x{config.IMG_SIZE}")
    print(f"Train/Val/Test split: {config.TRAIN_SPLIT}/{config.VAL_SPLIT}/{config.TEST_SPLIT}")
    print(f"Models to compare: {config.MODELS_TO_COMPARE}")

    # ------------------------------------------------------------
    # 1-10. Data preparation (shared across all models)
    # ------------------------------------------------------------
    data = prepare_data()

    # ------------------------------------------------------------
    # 11. Train + evaluate every model in config.MODELS_TO_COMPARE
    # ------------------------------------------------------------
    results = {}
    for model_name in config.MODELS_TO_COMPARE:
        results[model_name] = run_model_pipeline(model_name, data)

    # ------------------------------------------------------------
    # 12. Compare all trained models side-by-side
    # ------------------------------------------------------------
    compare_models.run_full_comparison(results)


if __name__ == "__main__":
    main()
