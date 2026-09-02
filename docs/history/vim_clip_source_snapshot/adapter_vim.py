"""Adapter utilities for Vim visual backbones.

Adapters are inserted after selected Vim backbone layers. They add a small
bottleneck residual MLP on the hidden states while leaving the original layer
path intact.
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union

import torch
import torch.nn as nn


SUPPORTED_ADAPTER_LAYERS = {"all", "last12", "back12", "front12", "first12"}


class VimAdapter(nn.Module):
    """Bottleneck residual adapter for Vim hidden states."""

    def __init__(
        self,
        feature_dim: int,
        adapter_dim: int = 64,
        dropout: float = 0.1,
        scale: float = 1.0,
        zero_init: bool = True,
    ):
        super().__init__()
        if adapter_dim <= 0:
            raise ValueError(f"adapter_dim must be positive, got {adapter_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"adapter dropout must be in [0, 1), got {dropout}")

        self.feature_dim = int(feature_dim)
        self.adapter_dim = int(adapter_dim)
        self.scale = float(scale)
        self.norm = nn.LayerNorm(self.feature_dim)
        self.down = nn.Linear(self.feature_dim, self.adapter_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.up = nn.Linear(self.adapter_dim, self.feature_dim)

        if zero_init:
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.scale * self.up(self.dropout(self.act(self.down(self.norm(x)))))


class VimLayerWithAdapter(nn.Module):
    """Wrap one Vim block and apply an adapter to its tensor output."""

    def __init__(self, base_layer: nn.Module, adapter: VimAdapter):
        super().__init__()
        self.base_layer = base_layer
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        out = self.base_layer(*args, **kwargs)
        if torch.is_tensor(out):
            return self.adapter(out)
        if isinstance(out, tuple) and out and torch.is_tensor(out[0]):
            return (self.adapter(out[0]), *out[1:])
        return out


def get_vim_layers(vim_or_visual) -> nn.ModuleList:
    """Return the Vim backbone layers ModuleList from a backbone or VimVisionWrapper."""
    backbone = getattr(vim_or_visual, "backbone", vim_or_visual)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise ValueError("Vim backbone does not expose a .layers ModuleList.")
    if not isinstance(layers, nn.ModuleList):
        raise TypeError(f"Vim backbone .layers must be nn.ModuleList, got {type(layers)!r}")
    return layers


def parse_adapter_layer_indices(layers: Union[str, Sequence[int]], total_layers: int) -> List[int]:
    if isinstance(layers, str):
        spec = layers.strip().lower()
        if spec in {"all"}:
            return list(range(total_layers))
        if spec in {"last12", "back12"}:
            start = max(0, total_layers - 12)
            return list(range(start, total_layers))
        if spec in {"front12", "first12"}:
            end = min(12, total_layers)
            return list(range(0, end))
        if ":" in spec:
            start_s, end_s = spec.split(":", 1)
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else total_layers
            return list(range(max(0, start), min(total_layers, end)))
        if "," in spec:
            return [idx for idx in (int(part.strip()) for part in spec.split(",") if part.strip()) if 0 <= idx < total_layers]
        try:
            idx = int(spec)
            return [idx] if 0 <= idx < total_layers else []
        except ValueError as exc:
            raise ValueError(
                f"Unsupported adapter layer spec: {layers!r}. Use all, last12, front12, 'start:end', or comma indices."
            ) from exc

    return [int(idx) for idx in layers if 0 <= int(idx) < total_layers]


def apply_adapters_to_vim(
    vim_or_visual,
    layers: Union[str, Sequence[int]] = "last12",
    feature_dim: int = 768,
    adapter_dim: int = 64,
    dropout: float = 0.1,
    scale: float = 1.0,
    zero_init: bool = True,
) -> List[int]:
    """Wrap selected Vim layers with residual bottleneck adapters."""
    vim_layers = get_vim_layers(vim_or_visual)
    layer_indices = parse_adapter_layer_indices(layers, len(vim_layers))
    if not layer_indices:
        raise ValueError(f"No Vim layers selected by adapter layer spec {layers!r}.")

    replacements = []
    for idx in layer_indices:
        layer = vim_layers[idx]
        if isinstance(layer, VimLayerWithAdapter):
            continue
        adapter = VimAdapter(
            feature_dim=feature_dim,
            adapter_dim=adapter_dim,
            dropout=dropout,
            scale=scale,
            zero_init=zero_init,
        )
        try:
            ref_param = next(layer.parameters())
            adapter = adapter.to(device=ref_param.device, dtype=ref_param.dtype)
        except StopIteration:
            pass
        vim_layers[idx] = VimLayerWithAdapter(layer, adapter)
        replacements.append(idx)
    return replacements


def iter_adapter_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for submodule in module.modules():
        if isinstance(submodule, VimAdapter):
            yield from submodule.parameters()


def mark_adapter_trainable(
    model: nn.Module,
    train_head: bool = True,
    train_backbone: str = "none",
):
    """Freeze all params, then enable adapters, optional head, and optional backbone scope.

    train_backbone:
        none: adapters only, plus optional head
        back12: adapters + last 12 original Vim layers, plus optional head
        full: adapters + full Vim backbone, plus optional head
    """
    for param in model.parameters():
        param.requires_grad = False

    logit_scale = getattr(model, "logit_scale", None)
    if logit_scale is not None:
        logit_scale.requires_grad = False

    visual = getattr(model, "visual", None)
    if visual is None:
        raise ValueError("Model does not expose .visual")

    if train_head:
        for head_name in ("ln_post", "proj"):
            head = getattr(visual, head_name, None)
            if head is None:
                continue
            if isinstance(head, nn.Parameter):
                head.requires_grad = True
            elif hasattr(head, "parameters"):
                for param in head.parameters():
                    param.requires_grad = True

    backbone_mode = str(train_backbone).lower()
    if backbone_mode not in {"none", "back12", "full"}:
        raise ValueError("train_backbone must be one of: none, back12, full")

    if backbone_mode == "full":
        backbone = getattr(visual, "backbone", None)
        if backbone is not None:
            for param in backbone.parameters():
                param.requires_grad = True
    elif backbone_mode == "back12":
        layers = list(get_vim_layers(visual))
        for layer in layers[max(0, len(layers) - 12):]:
            for param in layer.parameters():
                param.requires_grad = True

    for param in iter_adapter_parameters(visual):
        param.requires_grad = True


def split_adapter_head_backbone_parameters(model: nn.Module):
    visual = getattr(model, "visual", None)
    if visual is None:
        raise ValueError("Model does not expose .visual")

    adapter_params = [param for param in iter_adapter_parameters(visual) if param.requires_grad]
    adapter_ids = {id(param) for param in adapter_params}

    head_params = []
    for head_name in ("ln_post", "proj"):
        head = getattr(visual, head_name, None)
        if head is None:
            continue
        if isinstance(head, nn.Parameter):
            if head.requires_grad and id(head) not in adapter_ids:
                head_params.append(head)
        elif hasattr(head, "parameters"):
            head_params.extend(
                param for param in head.parameters()
                if param.requires_grad and id(param) not in adapter_ids
            )
    head_ids = {id(param) for param in head_params}

    backbone_params = [
        param for param in visual.parameters()
        if param.requires_grad and id(param) not in adapter_ids and id(param) not in head_ids
    ]
    return adapter_params, head_params, backbone_params


def count_adapter_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    params = list(iter_adapter_parameters(model))
    if trainable_only:
        params = [param for param in params if param.requires_grad]
    return sum(param.numel() for param in params)


@dataclass
class AdapterSummary:
    replacements: List[int]
    adapter_params: int
    trainable_adapter_params: int

    @property
    def num_replacements(self) -> int:
        return len(self.replacements)


def summarize_adapters(model: nn.Module, replacements: List[int]) -> AdapterSummary:
    return AdapterSummary(
        replacements=list(replacements),
        adapter_params=count_adapter_parameters(model, trainable_only=False),
        trainable_adapter_params=count_adapter_parameters(model, trainable_only=True),
    )
