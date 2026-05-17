"""
ResNet18 loading and layer-group configuration.

We group the layers into six "display blocks" for the heatmap and discriminative-LR sections:
- stem: conv1 + bn1
- layer1: the first residual block
- layer2: the second residual block
- layer3: the third residual block
- layer4: the fourth residual block
- fc: the final fully-connected head
"""

import copy
import torch.nn as nn
from torchvision import models


DISPLAY_BLOCKS = ["stem", "layer1", "layer2", "layer3", "layer4", "fc"]


DISPLAY_LABELS = {
    "stem": "Stem (conv1+bn1)",
    "layer1": "Layer 1",
    "layer2": "Layer 2",
    "layer3": "Layer 3",
    "layer4": "Layer 4",
    "fc": "Head (fc)",
}


DISC_GROUPS = {
    "early": ["stem", "layer1"],
    "mid": ["layer2", "layer3"],
    "late": ["layer4"],
    "head": ["fc"],
}


def load_pretrained_resnet18(num_classes):
    """
    Load ResNet18 with ImageNet weights and replace its 1000-class
    head with a fresh num_classes head. Returns the model.

    The new head has random weights — this is the standard setup
    when adapting to a new task.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    feature_dim = model.fc.in_features
    model.fc = nn.Linear(feature_dim, num_classes)
    return model


def get_block_modules(model):
    """
    Return a dict mapping display-block name to a list of nn.Module
    objects that make up that block. The 'stem' aggregates conv1 + bn1
    (maxpool has no parameters but we don't need it here).
    """
    return {
        "stem":   [model.conv1, model.bn1],
        "layer1": [model.layer1],
        "layer2": [model.layer2],
        "layer3": [model.layer3],
        "layer4": [model.layer4],
        "fc":     [model.fc],
    }


def get_block_parameters(model, block_name):
    """List the parameters in one display block."""
    blocks = get_block_modules(model)
    params = []
    for module in blocks[block_name]:
        params.extend(list(module.parameters()))
    return params


def block_param_count(model, block_name):
    """Total parameter count in one display block."""
    return sum(p.numel() for p in get_block_parameters(model, block_name))


def freeze_all(model):
    """Set requires_grad = False on every parameter."""
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_blocks(model, block_names):
    """Set requires_grad = True on parameters in the named blocks only."""
    for name in block_names:
        for p in get_block_parameters(model, name):
            p.requires_grad = True


def freeze_bn_stats_outside(model, allowed_blocks):
    """
    Put every BatchNorm layer NOT in the allowed display blocks
    into eval mode, freezing its running statistics. This is the
    BatchNorm subtlety from the study notes: requires_grad alone
    is not enough.

    `allowed_blocks` is a list of display block names whose BN
    stats are allowed to keep updating (typically the ones being
    fine-tuned).
    """
    # Map each parameter-bearing block to its top-level prefix in
    # named_modules. The fc block contains no BN so it never
    # appears here — but we still include it for symmetry.
    block_prefixes = {
        "stem": ("conv1", "bn1"),
        "layer1": ("layer1",),
        "layer2": ("layer2",),
        "layer3": ("layer3",),
        "layer4": ("layer4",),
        "fc": ("fc",),
    }
    allowed_prefixes = []
    for b in allowed_blocks:
        allowed_prefixes.extend(block_prefixes[b])

    for name, mod in model.named_modules():
        if isinstance(mod, nn.BatchNorm2d):
            allowed = any(name.startswith(p) for p in allowed_prefixes)
            if not allowed:
                mod.eval()


def snapshot_parameters(model):
    """
    Return a dict mapping parameter name -> a CLONED tensor of its
    current values. Used as the "before" snapshot to compute
    parameter changes after training.
    """
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def parameter_change_per_block(model, snapshot):
    """
    Given a "before" snapshot from snapshot_parameters(), compute
    how much each display block's parameters have changed in total.

    Returns a dict {block_name: float} with L2 norms of the change
    vector for that block, normalised by the parameter count so
    blocks of different sizes are comparable.

    The metric: ‖ θ_after − θ_before ‖_2 / sqrt(n_params)
    which is roughly "average per-parameter change magnitude".
    """
    block_modules = get_block_modules(model)
    # Build a name -> block lookup
    name_to_block = {}
    for block_name, modules in block_modules.items():
        for module in modules:
            for pname, _ in module.named_parameters(recurse=True):
                # We need the name as it appears in the parent model's
                # named_parameters(). torchvision modules use the same
                # naming, so we can reconstruct it.
                # We'll build this by walking the model directly below.
                pass

    # Simpler approach: walk model.named_parameters() and decide which
    # block each name belongs to by its prefix.
    block_for_name = {}
    for full_name, _ in model.named_parameters():
        if full_name.startswith("conv1") or full_name.startswith("bn1"):
            block_for_name[full_name] = "stem"
        elif full_name.startswith("layer1"):
            block_for_name[full_name] = "layer1"
        elif full_name.startswith("layer2"):
            block_for_name[full_name] = "layer2"
        elif full_name.startswith("layer3"):
            block_for_name[full_name] = "layer3"
        elif full_name.startswith("layer4"):
            block_for_name[full_name] = "layer4"
        elif full_name.startswith("fc"):
            block_for_name[full_name] = "fc"

    # Accumulate squared change and parameter count per block
    sq_change = {b: 0.0 for b in DISPLAY_BLOCKS}
    n_params = {b: 0 for b in DISPLAY_BLOCKS}
    for name, p in model.named_parameters():
        if name not in block_for_name:
            continue
        if name not in snapshot:
            continue
        block = block_for_name[name]
        diff = (p.detach() - snapshot[name]).flatten()
        sq_change[block] += float((diff ** 2).sum().item())
        n_params[block] += p.numel()

    # Normalised per-parameter change
    out = {}
    for b in DISPLAY_BLOCKS:
        if n_params[b] == 0:
            out[b] = 0.0
        else:
            out[b] = (sq_change[b] / n_params[b]) ** 0.5
    return out


def deepcopy_model(model):
    """Convenience: deep-copy a model so we can train one branch and keep another."""
    return copy.deepcopy(model)
