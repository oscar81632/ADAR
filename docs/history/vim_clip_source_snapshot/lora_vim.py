"""LoRA utilities for Vim visual backbones.

This module is intentionally independent from train_vim.py so experiments can
switch LoRA targets/layers without making the training script harder to read.
"""

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union

import torch
import torch.nn as nn


SUPPORTED_LORA_TARGETS = {"in_proj", "out_proj", "x_proj", "dt_proj"}
SUPPORTED_LORA_LAYERS = {"all", "last12", "back12", "front12", "first12"}


class LoRALinear(nn.Module):
    """Wrap an nn.Linear with a frozen base path plus a trainable LoRA update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)!r}")
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

        in_features = base.in_features
        out_features = base.out_features
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}

        self.lora_A = nn.Linear(in_features, self.rank, bias=False, **factory_kwargs)
        self.lora_B = nn.Linear(self.rank, out_features, bias=False, **factory_kwargs)
        self.reset_lora_parameters()

        for param in self.base.parameters():
            param.requires_grad = False

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.dropout(x)))

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def weight(self):
        # Some Mamba code paths read .weight directly instead of calling the
        # Linear module. Return the merged effective weight so LoRA still
        # participates in those paths, e.g. in_proj and dt_proj.
        lora_delta = self.lora_B.weight @ self.lora_A.weight
        return self.base.weight + self.scaling * lora_delta

    @property
    def bias(self):
        return self.base.bias


def parse_lora_targets(targets: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(targets, str):
        parsed = [target.strip() for target in targets.split(",") if target.strip()]
    else:
        parsed = [str(target).strip() for target in targets if str(target).strip()]

    invalid = [target for target in parsed if target not in SUPPORTED_LORA_TARGETS]
    if invalid:
        raise ValueError(
            f"Unsupported LoRA target(s): {invalid}. Supported targets: {sorted(SUPPORTED_LORA_TARGETS)}"
        )
    if not parsed:
        raise ValueError("At least one LoRA target is required.")
    return parsed


def get_vim_layers(vim_or_visual) -> List[nn.Module]:
    """Return the Vim backbone layers from a backbone or VimVisionWrapper."""
    backbone = getattr(vim_or_visual, "backbone", vim_or_visual)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise ValueError("Vim backbone does not expose a .layers ModuleList.")
    return list(layers)


def parse_lora_layer_indices(layers: Union[str, Sequence[int]], total_layers: int) -> List[int]:
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
                f"Unsupported LoRA layer spec: {layers!r}. Use all, last12, front12, 'start:end', or comma indices."
            ) from exc

    return [int(idx) for idx in layers if 0 <= int(idx) < total_layers]


def _replace_module_attr(parent: nn.Module, attr_name: str, module: nn.Module):
    if not hasattr(parent, attr_name):
        raise ValueError(f"Module {parent.__class__.__name__} has no attribute {attr_name!r}")
    setattr(parent, attr_name, module)


def apply_lora_to_vim(
    vim_or_visual,
    targets: Union[str, Sequence[str]] = "out_proj",
    layers: Union[str, Sequence[int]] = "last12",
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> List[Tuple[int, str]]:
    """Insert LoRA modules into Vim mixer projections.

    Args:
        vim_or_visual: Either VimVisionWrapper or its .backbone module.
        targets: Comma string/list from {"in_proj", "out_proj", "x_proj", "dt_proj"}.
        layers: all, last12/back12, front12/first12, start:end, comma indices, or int list.
        rank: LoRA rank.
        alpha: LoRA scaling alpha. Effective scale is alpha / rank.
        dropout: Dropout applied before LoRA A.

    Returns:
        A list of (layer_index, target_name) replacements.
    """
    target_names = parse_lora_targets(targets)
    vim_layers = get_vim_layers(vim_or_visual)
    layer_indices = parse_lora_layer_indices(layers, len(vim_layers))
    if not layer_indices:
        raise ValueError(f"No Vim layers selected by LoRA layer spec {layers!r}.")

    replacements = []
    for idx in layer_indices:
        layer = vim_layers[idx]
        mixer = getattr(layer, "mixer", None)
        if mixer is None:
            raise ValueError(f"Vim layer {idx} does not expose .mixer")
        for target_name in target_names:
            target_module = getattr(mixer, target_name, None)
            if target_module is None:
                raise ValueError(f"Vim layer {idx}.mixer has no target {target_name!r}")
            if isinstance(target_module, LoRALinear):
                continue
            if not isinstance(target_module, nn.Linear):
                raise TypeError(
                    f"Vim layer {idx}.mixer.{target_name} must be nn.Linear, got {type(target_module)!r}"
                )
            _replace_module_attr(mixer, target_name, LoRALinear(target_module, rank, alpha, dropout))
            # Mamba's fused fast path reads projection .weight tensors directly,
            # which bypasses LoRALinear.forward(). Disable it so LoRA updates are
            # actually used during fine-tuning.
            if hasattr(mixer, "use_fast_path"):
                mixer.use_fast_path = False
            replacements.append((idx, target_name))
    return replacements


def iter_lora_parameters(module: nn.Module) -> Iterable[nn.Parameter]:
    for submodule in module.modules():
        if isinstance(submodule, LoRALinear):
            yield from submodule.lora_A.parameters()
            yield from submodule.lora_B.parameters()


def mark_only_lora_and_head_trainable(model: nn.Module, train_head: bool = True):
    """Freeze everything except LoRA params and optionally Vim ln_post/proj head."""
    for param in model.parameters():
        param.requires_grad = False

    logit_scale = getattr(model, "logit_scale", None)
    if logit_scale is not None:
        logit_scale.requires_grad = False

    visual = getattr(model, "visual", None)
    if visual is None:
        raise ValueError("Model does not expose .visual")

    for param in iter_lora_parameters(visual):
        param.requires_grad = True

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


def split_lora_and_head_parameters(model: nn.Module):
    visual = getattr(model, "visual", None)
    if visual is None:
        raise ValueError("Model does not expose .visual")

    lora_params = [param for param in iter_lora_parameters(visual) if param.requires_grad]
    lora_ids = {id(param) for param in lora_params}

    head_params = []
    for head_name in ("ln_post", "proj"):
        head = getattr(visual, head_name, None)
        if head is None:
            continue
        if isinstance(head, nn.Parameter):
            if head.requires_grad and id(head) not in lora_ids:
                head_params.append(head)
        elif hasattr(head, "parameters"):
            head_params.extend(
                param for param in head.parameters()
                if param.requires_grad and id(param) not in lora_ids
            )
    return lora_params, head_params


def count_lora_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    params = list(iter_lora_parameters(model))
    if trainable_only:
        params = [param for param in params if param.requires_grad]
    return sum(param.numel() for param in params)


@dataclass
class LoRASummary:
    replacements: List[Tuple[int, str]]
    lora_params: int
    trainable_lora_params: int

    @property
    def num_replacements(self) -> int:
        return len(self.replacements)


def summarize_lora(model: nn.Module, replacements: List[Tuple[int, str]]) -> LoRASummary:
    return LoRASummary(
        replacements=list(replacements),
        lora_params=count_lora_parameters(model, trainable_only=False),
        trainable_lora_params=count_lora_parameters(model, trainable_only=True),
    )
