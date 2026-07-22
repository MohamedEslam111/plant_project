"""
Model module for the Plant Health Checker project.

Builds pretrained CNN models (ImageNet weights) with the final
classification layer replaced to match the number of plant disease classes.

Four architectures are supported for comparison:
- ResNet50            : the main model used for this project
- DenseNet121         : comparison model
- MobileNetV3-Large   : comparison model (lightweight, mobile-friendly)
- ConvNeXt-Tiny       : comparison model (modern conv architecture)
"""

import torch.nn as nn
from torchvision import models

import config


def build_resnet50(num_classes: int):
    """
    Load a pretrained ResNet50 and replace its final layer for transfer learning.

    Args:
        num_classes: number of output classes for the new classification head.

    Returns:
        model: ResNet50 model moved to the configured device.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model = model.to(config.DEVICE)
    return model


def build_densenet121(num_classes: int):
    """
    Load a pretrained DenseNet121 and replace its final layer for transfer learning.

    Args:
        num_classes: number of output classes for the new classification head.

    Returns:
        model: DenseNet121 model moved to the configured device.
    """
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

    model = model.to(config.DEVICE)
    return model


def build_mobilenet_v3_large(num_classes: int):
    """
    Load a pretrained MobileNetV3-Large and replace its final layer for transfer learning.

    Args:
        num_classes: number of output classes for the new classification head.

    Returns:
        model: MobileNetV3-Large model moved to the configured device.
    """
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)

    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, num_classes)

    model = model.to(config.DEVICE)
    return model


def build_convnext_tiny(num_classes: int):
    """
    Load a pretrained ConvNeXt-Tiny and replace its final layer for transfer learning.

    Args:
        num_classes: number of output classes for the new classification head.

    Returns:
        model: ConvNeXt-Tiny model moved to the configured device.
    """
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, num_classes)

    model = model.to(config.DEVICE)
    return model


# Registry mapping model names to their builder functions.
# Add new architectures here to make them available for comparison.
MODEL_REGISTRY = {
    'resnet50': build_resnet50,
    'densenet121': build_densenet121,
    'mobilenet_v3_large': build_mobilenet_v3_large,
    'convnext_tiny': build_convnext_tiny,
}


def build_model(model_name: str, num_classes: int):
    """
    Build a model by name using the MODEL_REGISTRY.

    Args:
        model_name: one of the keys in MODEL_REGISTRY
            (e.g. 'resnet50', 'densenet121', 'mobilenet_v3_large', 'convnext_tiny').
        num_classes: number of output classes for the new classification head.

    Returns:
        model: the requested model, moved to the configured device.
    """
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

    builder_fn = MODEL_REGISTRY[model_name]
    return builder_fn(num_classes)
