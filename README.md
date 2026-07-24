<div align="center">

# 🌿 Plant Health Checker

**Deep Learning pipeline for plant disease classification — data preparation, multi-model training, and side-by-side comparison.**

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Try_it_Now-2ea44f?style=for-the-badge)](https://mohamedeslam111.github.io/plant-health-web)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat)](#license)

**[🚀 Live Demo](https://mohamedeslam111.github.io/plant-health-web)** · [Project Structure](#-project-structure) · [Setup](#-setup) · [Outputs](#-outputs)

</div>

---

## 📖 Overview

This project loads, visualizes, and preprocesses the **PlantVillage** plant disease dataset, then trains and compares four transfer-learning architectures to find the best model for automated plant disease detection:

| Model | Highlight |
|---|---|
| 🏆 **ResNet50** | Main project model |
| **DenseNet121** | Dense connections, strong baseline |
| **MobileNetV3-Large** | Lightweight — mobile/edge-deployment friendly |
| **ConvNeXt-Tiny** | Modern convolutional architecture |

A trained version of the model powers the **[live web demo](https://mohamedeslam111.github.io/plant-health-web)**, where you can upload a leaf image and get an instant health diagnosis.

---

## 🗂 Project Structure

```
plant_health_checker/
├── config.py              # Paths, hyperparameters, per-model output paths
├── dataset.py              # Dataset loading, species extraction, train/val/test split
├── transforms_utils.py     # Preprocessing & data augmentation transforms
├── visualization.py        # Sample/species visualization, preprocessing comparison
├── model.py                # Model definitions (MODEL_REGISTRY)
├── train.py                # Training loop, early stopping, checkpointing
├── evaluate.py              # Metrics, confusion matrix, prediction visualization
├── compare_models.py        # Side-by-side model comparison table, chart, report
├── main.py                  # Orchestrates the full pipeline — run this file
├── requirements.txt
└── README.md
```

---

## ⚙️ Pipeline Steps (run via `main.py`)

1. Load dataset and explore basic statistics
2. Extract plant species information from class names
3. Split dataset into Train (70%) / Val (15%) / Test (15%)
4. Visualize class distribution across splits
5. Visualize random samples and one sample per species
6. Define preprocessing/augmentation transforms
7. Apply preprocessing to each split
8. Create DataLoaders
9. Visualize preprocessed batches and compare with originals
10. Test DataLoader iteration
11. For **each model** in `config.MODELS_TO_COMPARE`:
    - Build the model (pretrained on ImageNet)
    - Train with early stopping + best-checkpoint saving
    - Evaluate on the test set: accuracy, precision, recall, F1, classification report, confusion matrix, top confusions, best/worst classes, and sample predictions
12. **Compare all models side-by-side**: results table, bar chart, overlaid training curves, and a written comparison report

---

## 🚀 Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set the dataset path** in `config.py` (or via the `PLANT_DATA_DIR` environment variable):
```python
DATA_DIR = 'path/to/your/PlantVillage/dataset'
```

Expected dataset structure:
```
PlantVillage/
├── Apple___Apple_scab/
├── Apple___Black_rot/
├── Tomato___Late_blight/
└── ...
```

> **Running on Google Colab?** Mount your Drive first, then set `PLANT_DATA_DIR` or edit `DATA_DIR` directly:
> ```python
> from google.colab import drive
> drive.mount('/content/drive')
> ```

**3. (Optional) Choose which models to compare** in `config.py`:
```python
MODELS_TO_COMPARE = ['resnet50', 'densenet121', 'mobilenet_v3_large', 'convnext_tiny']
```

**4. Run the full pipeline**
```bash
python main.py
```

---

## ➕ Adding Another Model to the Comparison

1. Open `model.py` and write a `build_<name>(num_classes)` function that loads a pretrained model and replaces its final classification layer.
2. Register it in `MODEL_REGISTRY`:
   ```python
   MODEL_REGISTRY = {
       'resnet50': build_resnet50,
       'densenet121': build_densenet121,
       'mobilenet_v3_large': build_mobilenet_v3_large,
       'convnext_tiny': build_convnext_tiny,
       'efficientnet_b0': build_efficientnet_b0,  # new entry
   }
   ```
3. Add the name to `config.MODELS_TO_COMPARE`. `main.py` will train, evaluate, and include it in the comparison automatically.

---

## 📊 Outputs

Results are saved to `./plant_project_results/`, with one subfolder per model so nothing overwrites:

```
plant_project_results/
├── resnet50/
│   ├── best_model.pth
│   ├── training_log.csv
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── prediction_samples.png
├── densenet121/
│   └── ... (same files)
├── mobilenet_v3_large/
│   └── ... (same files)
├── convnext_tiny/
│   └── ... (same files)
└── comparison/
    ├── model_comparison.csv          # side-by-side metrics table
    ├── model_comparison.png          # bar chart of key metrics
    ├── model_comparison_curves.png   # overlaid val loss/accuracy curves
    └── comparison_report.txt          # written summary + best model per metric
```

---

## 📈 Comparison Metrics

The final comparison (`compare_models.py`) reports, for each model:

| Metric | Why It Matters |
|---|---|
| Test Accuracy / Macro F1 / Weighted F1 | Overall classification quality |
| Macro Precision / Recall | Per-class performance, unweighted |
| Best Val Accuracy / Loss / Epoch | Training convergence behavior |
| Total Train Time (minutes) | Training cost |
| Avg Inference Time (ms/image) | Deployment / real-time suitability |
| Trainable Parameters (millions) | Model size / complexity |

---

## 🔧 Configuration

Key hyperparameters (in `config.py`):

| Parameter | Default |
|---|---|
| `BATCH_SIZE` | 32 |
| `IMG_SIZE` | 224 |
| `LEARNING_RATE` | 0.0001 |
| `NUM_EPOCHS` | 10 |
| `EARLY_STOP_PATIENCE` | 3 |
| `MODELS_TO_COMPARE` | `['resnet50', 'densenet121', 'mobilenet_v3_large', 'convnext_tiny']` |

---

<div align="center">

### 🌐 [Try the Live Demo →](https://mohamedeslam111.github.io/plant-health-web)

Made with 🌱 by [Mohamed Eslam](https://github.com/MohamedEslam111)

</div>
