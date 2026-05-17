"""
Dataset loading for the transfer-learning simulator. Two datasets:

* STL-10  - natural images, similar to ImageNet (similar domain)
* EuroSAT - satellite imagery, visually unlike ImageNet (different domain)
* MNIST - handwritten digits, nothing like ImageNet (extreme domain shift)

Both are loaded through torchvision and downloaded on first use. We
expose a single uniform interface that returns a subset of N images
chosen deterministically by seed, so the same N value gives the same
images across runs.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

DATA_ROOT = Path.home() / "data"
DATA_ROOT.mkdir(exist_ok=True)


# ImageNet preprocessing since the pretrained ResNet18 expects inputs of size 224x224 normalised with ImageNet statistics.
IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _load_stl10():
    train = datasets.STL10(
        root=str(DATA_ROOT), split="train", download=True,
        transform=IMAGENET_TRANSFORM,
    )
    test = datasets.STL10(
        root=str(DATA_ROOT), split="test", download=True,
        transform=IMAGENET_TRANSFORM,
    )
    return train, test, list(train.classes)


def _load_eurosat():
    """
    EuroSAT comes as a single dataset. It is split into train/test
    deterministically. About 27,000 images across 10 land-use classes.
    """
    full = datasets.EuroSAT(
        root=str(DATA_ROOT), download=True,
        transform=IMAGENET_TRANSFORM,
    )
    # Deterministic split of first 80% train, last 20% test
    n = len(full)
    split = int(0.8 * n)
    indices = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    train = Subset(full, indices[:split].tolist())
    test = Subset(full, indices[split:].tolist())
    return train, test, list(full.classes)


def _load_mnist():
    """
    MNIST handwritten digits - 10 classes, grayscale, 28×28 pixels.
    This is the most drastic domain shift from ImageNet. The pretrained ResNet18.
    
    We replicate the single grayscale channel to RGB (3×224×224) so
    ResNet18's input layer is satisfied.
    """
    mnist_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225],
        ),
    ])
    train = datasets.MNIST(
        root=str(DATA_ROOT), train=True, download=True,
        transform=mnist_transform,
    )
    test = datasets.MNIST(
        root=str(DATA_ROOT), train=False, download=True,
        transform=mnist_transform,
    )
    return train, test, [str(i) for i in range(10)]


DATASETS = {
    "STL-10 (similar to ImageNet)": _load_stl10,
    "EuroSAT (drastically different)": _load_eurosat,
    "MNIST (nothing like ImageNet)": _load_mnist,
}


# Cached datasets - loading is slow on first run, so cache by name.
_dataset_cache = {}


def get_dataset(name):
    """Return (train_subset_pool, test_subset_pool, class_names) for the named dataset."""
    if name not in _dataset_cache:
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name}")
        _dataset_cache[name] = DATASETS[name]()
    return _dataset_cache[name]


def take_subset(dataset, n_samples, seed=0):
    """
    Return a Subset of `dataset` containing exactly `n_samples` items,
    chosen deterministically by `seed`. Used to give the user
    fine-grained control over how much data the model gets to fine-tune on.
    """
    n = len(dataset)
    n_samples = min(n_samples, n)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n, size=n_samples, replace=False)
    return Subset(dataset, chosen.tolist())


def get_dataset_info():
    """Return a dict of {name: short description} for UI display."""
    return {
        "STL-10 (similar to ImageNet)": (
            "10 classes of natural photos (animals, vehicles). 96x96 images "
            "upscaled to 224x224. Visually close to ImageNet - pretrained "
            "features should transfer well."
        ),
        "EuroSAT (drastically different)": (
            "10 land-use classes from satellite imagery (forest, residential, "
            "river, etc). Top-down aerial view, very different from ImageNet's "
            "ground-level photographs. Pretrained features need substantial "
            "adaptation."
        ),
        "MNIST (nothing like ImageNet)": (
            "Handwritten digit recognition with  10 classes (0–9). Grayscale 28x28 "
            "images converted to 3-channel 224x224.The pretrained backbone (ResNet18) has never seen "
            "anything like this. Worst-case transfer scenario."
        ),
    }
