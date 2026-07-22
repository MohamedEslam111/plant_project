"""
Evaluation module for the Plant Health Checker project.

Handles:
- Plotting training/validation curves
- Running inference on the test set
- Computing accuracy, precision, recall, F1-score (macro & weighted)
- Generating a full classification report
- Plotting a normalized confusion matrix
- Listing the top confused class pairs
- Listing best/worst performing classes by F1-score
- Visualizing correct vs incorrect predictions
"""

import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

import config
from transforms_utils import denormalize


def count_parameters(model) -> int:
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_curves(history, model_name):
    """Plot and save training/validation loss and accuracy curves for one model."""
    paths = config.get_model_paths(model_name)

    training_losses = history['train_losses']
    training_accuracies = history['train_accuracies']
    validation_losses = history['val_losses']
    validation_accuracies = history['val_accuracies']

    epochs_range = range(1, len(training_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(epochs_range, training_losses, 'o-', label='Train Loss', color='#2563eb')
    axes[0].plot(epochs_range, validation_losses, 'o-', label='Val Loss', color='#dc2626')
    axes[0].set_title(f'{model_name} - Training & Validation Loss', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy curve
    axes[1].plot(epochs_range, training_accuracies, 'o-', label='Train Acc', color='#2563eb')
    axes[1].plot(epochs_range, validation_accuracies, 'o-', label='Val Acc', color='#dc2626')
    axes[1].set_title(f'{model_name} - Training & Validation Accuracy', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(paths['training_curves'], dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Best Val Accuracy: {max(validation_accuracies):.2f}% "
          f"(Epoch {validation_accuracies.index(max(validation_accuracies)) + 1})")
    print(f"Best Val Loss: {min(validation_losses):.4f} "
          f"(Epoch {validation_losses.index(min(validation_losses)) + 1})")


def run_inference_on_test_set(model, test_loader):
    """
    Run the trained model on the test set and collect predictions.

    Returns:
        all_preds, all_labels: numpy arrays of predicted and true labels.
        avg_inference_time_ms: average inference time per image, in milliseconds.
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating on Test Set"):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE).long()

            start = time.time()
            if config.DEVICE.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
            total_time += time.time() - start
            total_images += images.size(0)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_inference_time_ms = (total_time / total_images) * 1000

    print(f"Number of test samples: {len(all_labels)}")
    print(f"Average inference time: {avg_inference_time_ms:.2f} ms/image")
    return all_preds, all_labels, avg_inference_time_ms


def compute_overall_metrics(all_labels, all_preds):
    """Compute and print accuracy, precision, recall, and F1 (macro & weighted)."""
    test_accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print("=" * 60)
    print("Final Evaluation Results on Test Set")
    print("=" * 60)
    print(f"Accuracy           : {test_accuracy * 100:.2f}%")
    print()
    print("Macro Average (unweighted across classes):")
    print(f"  Precision        : {precision_macro * 100:.2f}%")
    print(f"  Recall           : {recall_macro * 100:.2f}%")
    print(f"  F1-Score         : {f1_macro * 100:.2f}%")
    print()
    print("Weighted Average (weighted by class sample count):")
    print(f"  Precision        : {precision_weighted * 100:.2f}%")
    print(f"  Recall           : {recall_weighted * 100:.2f}%")
    print(f"  F1-Score         : {f1_weighted * 100:.2f}%")
    print("=" * 60)

    metrics = {
        'accuracy': test_accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
    }
    return metrics


def save_classification_report(all_labels, all_preds, class_names, metrics, model_name):
    """Generate, print, and save a full per-class classification report."""
    paths = config.get_model_paths(model_name)

    print("\nClassification Report per class:\n")
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0
    )
    print(report)

    with open(paths['report'], 'w', encoding='utf-8') as f:
        f.write(f"Plant Health Checker - Evaluation Report ({model_name})\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%\n")
        f.write(f"Macro F1: {metrics['f1_macro'] * 100:.2f}% | "
                f"Weighted F1: {metrics['f1_weighted'] * 100:.2f}%\n\n")
        f.write(report)

    print(f"\nReport saved to: {paths['report']}")


def plot_confusion_matrix(all_labels, all_preds, class_names, model_name):
    """Plot and save a normalized confusion matrix as a heatmap."""
    paths = config.get_model_paths(model_name)

    cm = confusion_matrix(all_labels, all_preds)

    num_classes_in_cm = len(class_names)
    figsize = (max(12, num_classes_in_cm * 0.4), max(10, num_classes_in_cm * 0.35))

    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm_normalized,
        annot=False,  # Set to True if the number of classes is small (<20)
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion'}
    )
    plt.title(f'{model_name} - Confusion Matrix (Normalized) - Test Set', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(paths['confusion_matrix'], dpi=150, bbox_inches='tight')
    plt.show()

    return cm


def print_top_confusions(cm, class_names, top_n=10):
    """Print the most frequent misclassification pairs (True -> Predicted)."""
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)  # Ignore correct predictions on the diagonal

    confusion_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm_copy[i, j] > 0:
                confusion_pairs.append((class_names[i], class_names[j], cm_copy[i, j]))

    confusion_pairs.sort(key=lambda x: x[2], reverse=True)

    print(f"Top {top_n} confused class pairs (True -> Predicted):\n")
    for true_cls, pred_cls, count in confusion_pairs[:top_n]:
        print(f"  {true_cls:40s} -> {pred_cls:40s} : {count} cases")


def print_best_worst_classes(all_labels, all_preds, class_names, top_n=5):
    """Print the best and worst performing classes ranked by F1-score."""
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)

    class_scores = list(zip(class_names, per_class_f1, per_class_precision, per_class_recall))
    class_scores_sorted = sorted(class_scores, key=lambda x: x[1])

    print(f"Worst {top_n} classes (lowest F1-Score):\n")
    print(f"{'Class':45s} {'F1':>8s} {'Precision':>10s} {'Recall':>8s}")
    for name, f1, prec, rec in class_scores_sorted[:top_n]:
        print(f"{name:45s} {f1 * 100:7.2f}% {prec * 100:9.2f}% {rec * 100:7.2f}%")

    print(f"\nBest {top_n} classes (highest F1-Score):\n")
    print(f"{'Class':45s} {'F1':>8s} {'Precision':>10s} {'Recall':>8s}")
    for name, f1, prec, rec in class_scores_sorted[-top_n:][::-1]:
        print(f"{name:45s} {f1 * 100:7.2f}% {prec * 100:9.2f}% {rec * 100:7.2f}%")


def visualize_prediction_samples(model, test_loader, class_names, model_name):
    """Visualize 4 correct and 4 incorrect predictions from the test set."""
    paths = config.get_model_paths(model_name)

    correct_samples = []
    wrong_samples = []

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images_gpu = images.to(config.DEVICE)
            labels_gpu = labels.to(config.DEVICE).long()

            if config.DEVICE.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(images_gpu)
            else:
                outputs = model(images_gpu)

            _, predicted = torch.max(outputs, 1)

            for img, true_l, pred_l in zip(images, labels, predicted.cpu()):
                if true_l.item() == pred_l.item() and len(correct_samples) < 4:
                    correct_samples.append((img, true_l.item(), pred_l.item()))
                elif true_l.item() != pred_l.item() and len(wrong_samples) < 4:
                    wrong_samples.append((img, true_l.item(), pred_l.item()))

            if len(correct_samples) >= 4 and len(wrong_samples) >= 4:
                break

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, (img, true_l, pred_l) in enumerate(correct_samples):
        ax = axes[0, i]
        ax.imshow(denormalize(img).permute(1, 2, 0).numpy())
        ax.set_title(f"True: {class_names[true_l]}\nPred: {class_names[pred_l]}",
                     fontsize=8, color='green')
        ax.axis('off')
    axes[0, 0].set_ylabel('Correct', fontsize=12, fontweight='bold')

    for i, (img, true_l, pred_l) in enumerate(wrong_samples):
        ax = axes[1, i]
        ax.imshow(denormalize(img).permute(1, 2, 0).numpy())
        ax.set_title(f"True: {class_names[true_l]}\nPred: {class_names[pred_l]}",
                     fontsize=8, color='red')
        ax.axis('off')
    axes[1, 0].set_ylabel('Wrong', fontsize=12, fontweight='bold')

    plt.suptitle(f'{model_name} - Sample Predictions on Test Set (Correct vs Wrong)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(paths['prediction_samples'], dpi=150, bbox_inches='tight')
    plt.show()


def print_final_summary(total_images, num_classes, species_dict, train_dataset,
                         val_dataset, test_dataset, checkpoint, metrics, model_name):
    """Print a final, report-ready project summary for one model."""
    paths = config.get_model_paths(model_name)

    print("=" * 70)
    print(f" {model_name.upper()} - FINAL PROJECT SUMMARY ".center(70, "="))
    print("=" * 70)
    print(f"Dataset                 : PlantVillage")
    print(f"Total Images            : {total_images:,}")
    print(f"Number of Classes       : {num_classes}")
    print(f"Number of Plant Species : {len(species_dict)}")
    print(f"Model Architecture      : {model_name} (Transfer Learning, ImageNet weights)")
    print(f"Image Size              : {config.IMG_SIZE}x{config.IMG_SIZE}")
    print(f"Train/Val/Test Split    : {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}")
    print("-" * 70)
    print(f"Best Epoch (Val Loss)   : {checkpoint['epoch'] + 1}")
    print(f"Best Val Accuracy       : {checkpoint['val_acc']:.2f}%")
    print(f"Best Val Loss           : {checkpoint['val_loss']:.4f}")
    print("-" * 70)
    print(f"FINAL TEST ACCURACY     : {metrics['accuracy'] * 100:.2f}%")
    print(f"Test Macro F1-Score     : {metrics['f1_macro'] * 100:.2f}%")
    print(f"Test Weighted F1-Score  : {metrics['f1_weighted'] * 100:.2f}%")
    print(f"Test Macro Precision    : {metrics['precision_macro'] * 100:.2f}%")
    print(f"Test Macro Recall       : {metrics['recall_macro'] * 100:.2f}%")
    print("=" * 70)
    print(f"\nFiles saved in: {config.get_model_save_dir(model_name)}")
    print("  - best_model.pth             (best model weights)")
    print("  - training_log.csv          (training log per epoch)")
    print("  - training_curves.png       (loss/accuracy curves)")
    print("  - confusion_matrix.png      (confusion matrix)")
    print("  - classification_report.txt (detailed per-class report)")
    print("  - prediction_samples.png    (sample predictions)")
