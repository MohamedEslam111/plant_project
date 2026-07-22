"""
Configuration file for the Plant Health Checker project.
Contains dataset paths, hyperparameters, and training settings.
"""

import os
import torch

# ============================================================
# Reproducibility
# ============================================================
RANDOM_SEED = 42

# ============================================================
# Dataset path
# ============================================================
# Update this path to your dataset location.
# Expected structure:
# dataset/
# ├── Apple___Apple_scab/
# ├── Apple___Black_rot/
# ├── Tomato___Late_blight/
# └── ...
#
# If running on Google Colab, mount your Drive first, then either set
# the PLANT_DATA_DIR environment variable or edit the default path below:
#
#     from google.colab import drive
#     drive.mount('/content/drive')
#
DATA_DIR = os.environ.get('PLANT_DATA_DIR', '/content/drive/MyDrive/PlantVillage')

# ============================================================
# Hyperparameters
# ============================================================
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_WORKERS = 4

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

LEARNING_RATE = 0.0001
NUM_EPOCHS = 10
EARLY_STOP_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2

# ============================================================
# ImageNet normalization statistics (required for ResNet transfer learning)
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================
# Models to train and compare
# ============================================================
# Each entry is a model name available in model.MODEL_REGISTRY.
# The first model in the list is treated as the main project model.
MODELS_TO_COMPARE = ['resnet50', 'densenet121', 'mobilenet_v3_large', 'convnext_tiny']

# ============================================================
# Output paths
# ============================================================
# Base results directory, with one subfolder per model so that
# checkpoints/logs/plots never overwrite each other.
SAVE_DIR = './plant_project_results/'
COMPARISON_DIR = os.path.join(SAVE_DIR, 'comparison')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(COMPARISON_DIR, exist_ok=True)


def get_model_save_dir(model_name: str) -> str:
    """Return (and create) the results subfolder for a given model."""
    model_dir = os.path.join(SAVE_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def get_model_paths(model_name: str) -> dict:
    """
    Return all output file paths for a given model, namespaced under
    its own results subfolder (./plant_project_results/<model_name>/).
    """
    model_dir = get_model_save_dir(model_name)
    return {
        'csv_log': os.path.join(model_dir, 'training_log.csv'),
        'best_model': os.path.join(model_dir, 'best_model.pth'),
        'report': os.path.join(model_dir, 'classification_report.txt'),
        'training_curves': os.path.join(model_dir, 'training_curves.png'),
        'confusion_matrix': os.path.join(model_dir, 'confusion_matrix.png'),
        'prediction_samples': os.path.join(model_dir, 'prediction_samples.png'),
    }


# Comparison output paths (combined results across all models)
COMPARISON_TABLE_PATH = os.path.join(COMPARISON_DIR, 'model_comparison.csv')
COMPARISON_CHART_PATH = os.path.join(COMPARISON_DIR, 'model_comparison.png')
COMPARISON_CURVES_PATH = os.path.join(COMPARISON_DIR, 'model_comparison_curves.png')
COMPARISON_REPORT_PATH = os.path.join(COMPARISON_DIR, 'comparison_report.txt')

# ============================================================
# Device
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed across libraries for reproducible results."""
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
