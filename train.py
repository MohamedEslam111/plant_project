"""
Training module for the Plant Health Checker project.

Implements the full training loop with:
- Cross-entropy loss
- Adam optimizer (configured in main.py)
- ReduceLROnPlateau learning rate scheduler
- Early stopping based on validation loss
- Best-checkpoint saving
- CSV logging of metrics per epoch
"""

import csv
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

import config


def _init_csv_log(csv_path: str) -> None:
    """Create the CSV log file with headers if it doesn't already exist."""
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'LR'])


def _log_epoch_to_csv(csv_path: str, epoch: int, train_loss: float, train_acc: float,
                       val_loss: float, val_acc: float, lr: float) -> None:
    """Append one epoch's metrics to the CSV log."""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, lr])


def train_one_epoch(model, train_loader, criterion, optimizer, epoch, num_epochs):
    """Run a single training epoch and return (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    train_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] - Training")

    for images, labels in train_bar:
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, val_loader, criterion):
    """Run validation for a single epoch and return (avg_loss, accuracy)."""
    model.eval()
    val_correct = 0
    val_total = 0
    val_running_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE).long()

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total
    return val_loss, val_acc


def train_model(model, model_name, train_loader, val_loader, optimizer):
    """
    Run the full training loop with early stopping and checkpointing.

    Args:
        model: the model to train.
        model_name: string identifier (e.g. 'resnet50', 'densenet121',
            'mobilenet_v3_large', 'convnext_tiny'),
            used to namespace output files under their own results folder.
        train_loader, val_loader: DataLoaders for training and validation.
        optimizer: optimizer instance (e.g., Adam).

    Returns:
        history: dict with keys 'train_losses', 'train_accuracies',
                  'val_losses', 'val_accuracies', 'epoch_times', 'total_train_time'
        checkpoint: dict with the best saved checkpoint info.
    """
    paths = config.get_model_paths(model_name)

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE
    )

    _init_csv_log(paths['csv_log'])

    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []
    epoch_times = []

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\n{'=' * 60}")
    print(f"Training model: {model_name}")
    print(f"{'=' * 60}")

    total_start = time.time()

    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()

        epoch_loss, epoch_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, config.NUM_EPOCHS
        )
        training_losses.append(epoch_loss)
        training_accuracies.append(epoch_acc)

        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)
        validation_losses.append(val_loss)
        validation_accuracies.append(val_acc)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch [{epoch + 1}/{config.NUM_EPOCHS}] "
            f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% || "
            f"LR: {current_lr:.6f} || Time: {epoch_time:.1f}s"
        )

        _log_epoch_to_csv(paths['csv_log'], epoch + 1, epoch_loss, epoch_acc, val_loss, val_acc, current_lr)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_name': model_name,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, paths['best_model'])
            print(f"  -> Saved new best checkpoint with Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{config.EARLY_STOP_PATIENCE}")

        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered at Epoch {epoch + 1}")
            break

    total_train_time = time.time() - total_start

    # Load the best checkpoint after training completes
    checkpoint = torch.load(paths['best_model'], map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(
        f"\nLoaded best model from Epoch {checkpoint['epoch'] + 1} "
        f"with Val Loss: {checkpoint['val_loss']:.4f} and Val Acc: {checkpoint['val_acc']:.2f}%"
    )
    print(f"Total training time for {model_name}: {total_train_time / 60:.1f} minutes")

    history = {
        'train_losses': training_losses,
        'train_accuracies': training_accuracies,
        'val_losses': validation_losses,
        'val_accuracies': validation_accuracies,
        'epoch_times': epoch_times,
        'total_train_time': total_train_time,
    }

    return history, checkpoint
