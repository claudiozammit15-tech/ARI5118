"""
training.py
-----------
Real fine-tuning loops for the four techniques. 

Each function returns:
    model    - the trained model (a fresh copy where the input model is not modified)
    history  - dict with 'loss' and 'train_acc' lists, one entry per epoch

The four functions are:
    train_frozen_extraction   - only the head trains
    train_uniform_finetune    - every parameter trains, single LR
    train_gradual_unfreezing  - progressively unfreezes from output toward input
    train_discriminative      - every parameter trains, four LR groups
    train_lp_ft               - two-stage: linear probe, then full fine-tune

All loops put non-trainable BatchNorm layers into eval mode so their
running statistics don't drift during training (the BatchNorm
subtlety from the study notes).
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model_utils import (
    DISPLAY_BLOCKS,
    DISC_GROUPS,
    freeze_all,
    freeze_bn_stats_outside,
    get_block_parameters,
    unfreeze_blocks,
)


def _evaluate_acc(model, loader):
    """Top-1 accuracy on a DataLoader, in eval mode."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return correct / total if total > 0 else 0.0


def _train_one_epoch(model, loader, optimizer, loss_fn, allowed_bn_blocks):
    model.train()
    if allowed_bn_blocks is not None:
        freeze_bn_stats_outside(model, allowed_bn_blocks)

    losses, correct, total = [], 0, 0
    for x, y in loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    return sum(losses) / max(1, len(losses)), correct / max(1, total)


# Frozen feature extraction
def train_frozen_extraction(model, train_dataset, lr=1e-2, epochs=10,
                             batch_size=32, seed=0, progress_callback=None):
    
    torch.manual_seed(seed)
    model = copy.deepcopy(model)
    freeze_all(model)
    unfreeze_blocks(model, ["fc"])

    optimizer = optim.Adam(get_block_parameters(model, "fc"), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {"loss": [], "train_acc": []}
    for epoch in range(epochs):
        loss, acc = _train_one_epoch(model, loader, optimizer, loss_fn,
                                      allowed_bn_blocks=[])
        history["loss"].append(loss)
        history["train_acc"].append(acc)
        if progress_callback:
            progress_callback(epoch + 1, epochs, loss, acc)
    return model, history


# Uniform end to end fine-tuning

def train_uniform_finetune(model, train_dataset, lr=1e-4, epochs=5,
                            batch_size=32, seed=0, progress_callback=None):
    
    torch.manual_seed(seed)
    model = copy.deepcopy(model)
    for p in model.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {"loss": [], "train_acc": []}
    for epoch in range(epochs):
        loss, acc = _train_one_epoch(model, loader, optimizer, loss_fn,
                                      allowed_bn_blocks=DISPLAY_BLOCKS)
        history["loss"].append(loss)
        history["train_acc"].append(acc)
        if progress_callback:
            progress_callback(epoch + 1, epochs, loss, acc)
    return model, history


# Gradual unfreezing

def train_gradual_unfreezing(model, train_dataset, lr=1e-4,
                              epochs_per_stage=2, batch_size=32, seed=0,
                              progress_callback=None):
    """
    Five stages, working from output toward input:
        Stage 1: only fc trains
        Stage 2: fc + layer4 train
        Stage 3: + layer3
        Stage 4: + layer2
        Stage 5: + layer1 + stem (everything)

    Each stage runs for `epochs_per_stage` epochs. Total training
    is therefore 5 * epochs_per_stage epochs.
    """
    torch.manual_seed(seed)
    model = copy.deepcopy(model)
    freeze_all(model)

    stages = [
        ["fc"],
        ["fc", "layer4"],
        ["fc", "layer4", "layer3"],
        ["fc", "layer4", "layer3", "layer2"],
        ["fc", "layer4", "layer3", "layer2", "layer1", "stem"],
    ]

    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {"loss": [], "train_acc": [], "stage": []}
    total_epochs = len(stages) * epochs_per_stage
    epoch_counter = 0

    for stage_idx, unfrozen in enumerate(stages):
        freeze_all(model)
        unfreeze_blocks(model, unfrozen)

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable, lr=lr)

        for _ in range(epochs_per_stage):
            loss, acc = _train_one_epoch(model, loader, optimizer, loss_fn,
                                          allowed_bn_blocks=unfrozen)
            history["loss"].append(loss)
            history["train_acc"].append(acc)
            history["stage"].append(stage_idx + 1)
            epoch_counter += 1
            if progress_callback:
                progress_callback(epoch_counter, total_epochs, loss, acc)

    return model, history


# Discriminative learning rates
def train_discriminative(model, train_dataset,
                          lr_head=1e-2, lr_late=1e-3, lr_mid=1e-4, lr_early=1e-5,
                          epochs=5, batch_size=32, seed=0,
                          progress_callback=None):

    torch.manual_seed(seed)
    model = copy.deepcopy(model)
    for p in model.parameters():
        p.requires_grad = True

    # Build parameter groups by collecting params from each named block
    param_groups = []
    for group_name, lr in [
        ("head", lr_head),
        ("late", lr_late),
        ("mid", lr_mid),
        ("early", lr_early),
    ]:
        block_names = DISC_GROUPS[group_name]
        params_for_group = []
        for b in block_names:
            params_for_group.extend(get_block_parameters(model, b))
        if params_for_group:
            param_groups.append({"params": params_for_group, "lr": lr})

    optimizer = optim.Adam(param_groups)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {"loss": [], "train_acc": []}
    for epoch in range(epochs):
        # All blocks training, so allow all BN layers
        loss, acc = _train_one_epoch(model, loader, optimizer, loss_fn,
                                      allowed_bn_blocks=DISPLAY_BLOCKS)
        history["loss"].append(loss)
        history["train_acc"].append(acc)
        if progress_callback:
            progress_callback(epoch + 1, epochs, loss, acc)
    return model, history


# LP-FT
def train_lp_ft(model, train_dataset,
                 lp_lr=1e-2, lp_epochs=5,
                 ft_lr=1e-4, ft_epochs=3,
                 batch_size=32, seed=0, progress_callback=None):
   
    torch.manual_seed(seed)
    model = copy.deepcopy(model)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {"loss": [], "train_acc": [], "stage": []}
    total_epochs = lp_epochs + ft_epochs
    epoch_counter = 0

# train head only first
    freeze_all(model)
    unfreeze_blocks(model, ["fc"])
    head_optimizer = optim.Adam(get_block_parameters(model, "fc"), lr=lp_lr)

    for _ in range(lp_epochs):
        loss, acc = _train_one_epoch(model, loader, head_optimizer, loss_fn,
                                      allowed_bn_blocks=[])
        history["loss"].append(loss)
        history["train_acc"].append(acc)
        history["stage"].append("LP")
        epoch_counter += 1
        if progress_callback:
            progress_callback(epoch_counter, total_epochs, loss, acc)

# then unfreeze everything and fine-tune end to end
    for p in model.parameters():
        p.requires_grad = True
    full_optimizer = optim.Adam(model.parameters(), lr=ft_lr)

    for _ in range(ft_epochs):
        loss, acc = _train_one_epoch(model, loader, full_optimizer, loss_fn,
                                      allowed_bn_blocks=DISPLAY_BLOCKS)
        history["loss"].append(loss)
        history["train_acc"].append(acc)
        history["stage"].append("FT")
        epoch_counter += 1
        if progress_callback:
            progress_callback(epoch_counter, total_epochs, loss, acc)

    return model, history



def evaluate(model, dataset, batch_size=32, max_samples=None):
    if max_samples is not None and max_samples < len(dataset):
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(max_samples)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return _evaluate_acc(model, loader)
