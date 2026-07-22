# Plant Health Checker - Data Preparation, Training & Model Comparison

Loading, visualizing, and preprocessing the **PlantVillage** plant disease
dataset, then training and comparing four transfer-learning models:
**ResNet50** (main model), **DenseNet121**, **MobileNetV3-Large**, and
**ConvNeXt-Tiny**.

## Project Structure

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
├── main.py                  # Orchestrates the full pipeline, run this file
├── requirements.txt
└── README.md
```

## Models Compared

| Model               | Notes                                          |
|---------------------|-------------------------------------------------|
| ResNet50            | Main project model                              |
| DenseNet121         | Dense connections, strong baseline              |
| MobileNetV3-Large   | Lightweight, mobile/edge-deployment friendly    |
| ConvNeXt-Tiny       | Modern convolutional architecture               |

## Pipeline Steps (run via `main.py`)

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
    - Evaluate on the test set: accuracy, precision, recall, F1,
      classification report, confusion matrix, top confusions,
      best/worst classes, and sample predictions
12. **Compare all models side-by-side**: results table, bar chart,
    overlaid training curves, and a written comparison report

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Update the dataset path in `config.py` (or set the `PLANT_DATA_DIR`
   environment variable):
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

   **Running on Google Colab:** mount your Drive first, then either set
   the `PLANT_DATA_DIR` environment variable or edit `DATA_DIR` directly:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. (Optional) Change which models are compared in `config.py`:
   ```python
   MODELS_TO_COMPARE = ['resnet50', 'densenet121', 'mobilenet_v3_large', 'convnext_tiny']
   ```

4. Run the full pipeline:
   ```
   python main.py
   ```

## Adding another model to the comparison

Open `model.py` and:
1. Write a `build_<name>(num_classes)` function that loads a pretrained
   model and replaces its final classification layer.
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
3. Add the name to `config.MODELS_TO_COMPARE`. `main.py` will train,
   evaluate, and include it in the comparison automatically.

## Outputs

Results are saved to `./plant_project_results/`, with one subfolder
per model so nothing overwrites:

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

## Comparison Metrics

The final comparison (`compare_models.py`) reports, for each model:

| Metric                          | Why it matters                          |
|----------------------------------|------------------------------------------|
| Test Accuracy / Macro F1 / Weighted F1 | Overall classification quality      |
| Macro Precision / Recall         | Per-class performance, unweighted        |
| Best Val Accuracy / Loss / Epoch | Training convergence behavior            |
| Total Train Time (minutes)       | Training cost                            |
| Avg Inference Time (ms/image)    | Deployment / real-time suitability       |
| Trainable Parameters (millions)  | Model size / complexity                  |

## Configuration

Key hyperparameters (in `config.py`):

| Parameter         | Default  |
|-------------------|----------|
| `BATCH_SIZE`      | 32       |
| `IMG_SIZE`        | 224      |
| `LEARNING_RATE`   | 0.0001   |
| `NUM_EPOCHS`      | 10       |
| `EARLY_STOP_PATIENCE` | 3    |
| `MODELS_TO_COMPARE` | `['resnet50', 'densenet121', 'mobilenet_v3_large', 'convnext_tiny']` |
