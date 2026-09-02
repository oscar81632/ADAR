from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class ViTSuppOnlyReadout(nn.Module):
    """Clean complementary-evidence readout for CLIP ViT patch tokens."""

    def __init__(
        self,
        width: int,
        bottleneck: int = 128,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.wants_layerwise_tokens = False
        self.token_gate = nn.Sequential(OrderedDict([
            ("ln", LayerNorm(width)),
            ("down", nn.Linear(width, bottleneck)),
            ("act", QuickGELU()),
            ("up", nn.Linear(bottleneck, 1)),
        ]))
        self.router = nn.Sequential(OrderedDict([
            ("ln", LayerNorm(width)),
            ("down", nn.Linear(width, bottleneck)),
            ("act", QuickGELU()),
            ("up", nn.Linear(bottleneck, 1)),
        ]))
        self.fuse = nn.Sequential(OrderedDict([
            ("ln", LayerNorm(width * 2)),
            ("down", nn.Linear(width * 2, width)),
            ("act", QuickGELU()),
            ("drop", nn.Dropout(dropout)),
            ("up", nn.Linear(width, width)),
        ]))

    def gated_pool(self, patches: torch.Tensor):
        gates = torch.sigmoid(self.token_gate(patches)).squeeze(-1)
        weights = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("bn,bnd->bd", weights, patches)

    def forward(self, tokens: torch.Tensor, layerwise_tokens=None):
        cls = tokens[:, 0, :]
        patches = tokens[:, 1:, :]
        kept = self.gated_pool(patches)
        suppressed = patches.mean(dim=1) - kept
        alpha = torch.sigmoid(self.router(cls)).to(dtype=cls.dtype)
        pooled = torch.cat([cls, alpha * suppressed], dim=-1)
        return cls + self.residual_scale.to(dtype=cls.dtype) * self.fuse(pooled)


class ViTKeepOnlyReadout(nn.Module):
    """Clean selected-evidence readout for CLIP ViT patch tokens."""

    def __init__(
        self,
        width: int,
        bottleneck: int = 128,
        dropout: float = 0.1,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.wants_layerwise_tokens = False
        self.token_gate = nn.Sequential(OrderedDict([
            ("ln", LayerNorm(width)),
            ("down", nn.Linear(width, bottleneck)),
            ("act", QuickGELU()),
            ("up", nn.Linear(bottleneck, 1)),
        ]))
        self.router = nn.Sequential(OrderedDict([
            ("ln", LayerNorm(width)),
            ("down", nn.Linear(width, bottleneck)),
            ("act", QuickGELU()),
            ("up", nn.Linear(bottleneck, 1)),
        ]))
        self.fuse = nn.Sequential(OrderedDict([
            ("ln", LayerNorm(width * 2)),
            ("down", nn.Linear(width * 2, width)),
            ("act", QuickGELU()),
            ("drop", nn.Dropout(dropout)),
            ("up", nn.Linear(width, width)),
        ]))

    def gated_pool(self, patches: torch.Tensor):
        gates = torch.sigmoid(self.token_gate(patches)).squeeze(-1)
        weights = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("bn,bnd->bd", weights, patches)

    def forward(self, tokens: torch.Tensor, layerwise_tokens=None):
        cls = tokens[:, 0, :]
        patches = tokens[:, 1:, :]
        kept = self.gated_pool(patches)
        alpha = torch.sigmoid(self.router(cls)).to(dtype=cls.dtype)
        pooled = torch.cat([cls, alpha * kept], dim=-1)
        return cls + self.residual_scale.to(dtype=cls.dtype) * self.fuse(pooled)


class ViTPathReadout(nn.Module):
    """Path-aware readout heads for CLIP ViT patch tokens."""

    def __init__(
        self,
        width: int,
        grid_size: int,
        mode: str = "none",
        bottleneck: int = 128,
        dropout: float = 0.1,
        hops: int = 4,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        self.mode = mode
        self.grid_size = grid_size
        self.hops = hops
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.static_alpha = nn.Parameter(torch.tensor(0.0))
        self.wants_layerwise_tokens = mode in {"layerwise_decoy_aware", "geo_layer_mixed_adar"}

        self.start_query = nn.Parameter(torch.randn(width) * width ** -0.5)
        self.path_query = nn.Parameter(torch.randn(width) * width ** -0.5)
        self.path_queries = nn.Parameter(torch.randn(4, width) * width ** -0.5)
        self.direction_queries = nn.Parameter(torch.randn(4, width) * width ** -0.5)
        self.path_slots = nn.Parameter(torch.randn(4, width) * width ** -0.5)
        self.spatial_band_logits = nn.Parameter(torch.zeros(grid_size * grid_size))

        self.token_gate = nn.Sequential(OrderedDict([
            ("ln", LayerNorm(width)),
            ("down", nn.Linear(width, bottleneck)),
            ("act", QuickGELU()),
            ("up", nn.Linear(bottleneck, 1)),
        ]))
        self.multihead_gate = nn.Sequential(OrderedDict([
            ("ln", LayerNorm(width)),
            ("down", nn.Linear(width, bottleneck)),
            ("act", QuickGELU()),
            ("up", nn.Linear(bottleneck, 4)),
        ]))
        self.router = nn.Sequential(OrderedDict([
            ("ln", LayerNorm(width)),
            ("down", nn.Linear(width, bottleneck)),
            ("act", QuickGELU()),
            ("up", nn.Linear(bottleneck, 1)),
        ]))
        self.query_proj = nn.Linear(width, width)
        self.key_proj = nn.Linear(width, width)
        self.value_proj = nn.Linear(width, width)
        self.logit_router = nn.Sequential(OrderedDict([
            ("ln", LayerNorm(width)),
            ("down", nn.Linear(width, bottleneck)),
            ("act", QuickGELU()),
            ("up", nn.Linear(bottleneck, 1)),
        ]))
        self.slot_update = nn.GRUCell(width, width)
        self.layer_weights = nn.Parameter(torch.zeros(4))
        self.path_state = nn.Sequential(OrderedDict([
            ("dwconv", nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width)),
            ("pwdown", nn.Conv2d(width, bottleneck, kernel_size=1)),
            ("act", nn.GELU()),
            ("pwup", nn.Conv2d(bottleneck, width, kernel_size=1)),
        ]))
        self.hop_update = nn.GRUCell(width, width)
        self.fuse = nn.Sequential(OrderedDict([
            ("ln", LayerNorm(width * 3)),
            ("down", nn.Linear(width * 3, width)),
            ("act", QuickGELU()),
            ("drop", nn.Dropout(dropout)),
            ("up", nn.Linear(width, width)),
        ]))
        self.fuse_two = nn.Sequential(OrderedDict([
            ("ln", LayerNorm(width * 2)),
            ("down", nn.Linear(width * 2, width)),
            ("act", QuickGELU()),
            ("drop", nn.Dropout(dropout)),
            ("up", nn.Linear(width, width)),
        ]))
        if mode == "multihop_zero":
            nn.init.zeros_(self.fuse.up.weight)
            nn.init.zeros_(self.fuse.up.bias)

    def attention_pool(self, patches: torch.Tensor, query: torch.Tensor):
        query = query.to(dtype=patches.dtype, device=patches.device)
        scores = torch.einsum("bnd,d->bn", patches, query) / patches.shape[-1] ** 0.5
        weights = scores.softmax(dim=1)
        return torch.einsum("bn,bnd->bd", weights, patches), weights

    def cls_query_pool(self, patches: torch.Tensor, cls: torch.Tensor):
        query = self.query_proj(cls).unsqueeze(1)
        keys = self.key_proj(patches)
        values = self.value_proj(patches)
        scores = torch.einsum("bqd,bnd->bqn", query, keys) / patches.shape[-1] ** 0.5
        weights = scores.softmax(dim=-1)
        return torch.einsum("bqn,bnd->bqd", weights, values).squeeze(1)

    def propagate(self, patches: torch.Tensor):
        bsz, num_tokens, width = patches.shape
        grid = int(num_tokens ** 0.5)
        x = patches.transpose(1, 2).reshape(bsz, width, grid, grid)
        state = x + self.path_state(x)
        return state.flatten(2).transpose(1, 2)

    def gated_pool(self, patches: torch.Tensor):
        gates = torch.sigmoid(self.token_gate(patches)).squeeze(-1)
        weights = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.einsum("bn,bnd->bd", weights, patches)

    def complement_gated_pool(self, patches: torch.Tensor):
        gates = torch.sigmoid(self.token_gate(patches)).squeeze(-1).clamp(1e-4, 1.0 - 1e-4)
        keep_weights = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-6)
        comp_scores = 1.0 - gates
        comp_weights = comp_scores / comp_scores.sum(dim=1, keepdim=True).clamp_min(1e-6)
        kept = torch.einsum("bn,bnd->bd", keep_weights, patches)
        complement = torch.einsum("bn,bnd->bd", comp_weights, patches)
        return kept, complement

    def spatial_band_pool(self, patches: torch.Tensor):
        gates = torch.sigmoid(self.token_gate(patches)).squeeze(-1).clamp_min(1e-4)
        spatial = self.spatial_band_logits.to(dtype=patches.dtype, device=patches.device).softmax(dim=0)
        spatial = spatial.unsqueeze(0).expand_as(gates)
        keep_scores = gates * spatial
        keep_weights = keep_scores / keep_scores.sum(dim=1, keepdim=True).clamp_min(1e-6)
        comp_scores = (1.0 - gates).clamp_min(1e-4)
        comp_weights = comp_scores / comp_scores.sum(dim=1, keepdim=True).clamp_min(1e-6)
        kept = torch.einsum("bn,bnd->bd", keep_weights, patches)
        complement = torch.einsum("bn,bnd->bd", comp_weights, patches)
        return kept, complement

    def decoy_contrast_pool(self, patches: torch.Tensor):
        gates = torch.sigmoid(self.token_gate(patches)).squeeze(-1).clamp(1e-4, 1.0 - 1e-4)
        keep_weights = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-6)
        decoy_scores = 1.0 - gates
        decoy_weights = decoy_scores / decoy_scores.sum(dim=1, keepdim=True).clamp_min(1e-6)
        kept = torch.einsum("bn,bnd->bd", keep_weights, patches)
        decoy = torch.einsum("bn,bnd->bd", decoy_weights, patches)
        return kept, decoy

    def multihead_gated_pool(self, patches: torch.Tensor):
        gates = torch.sigmoid(self.multihead_gate(patches)).transpose(1, 2)
        weights = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        heads = torch.einsum("bhn,bnd->bhd", weights, patches)
        head_scores = torch.einsum("bhd,d->bh", heads, self.path_query.to(dtype=patches.dtype, device=patches.device))
        head_weights = head_scores.softmax(dim=-1)
        kept = torch.einsum("bh,bhd->bd", head_weights, heads)
        diversity = heads.mean(dim=1)
        return kept, diversity

    def directional_gated_pool(self, patches: torch.Tensor):
        direction_scores = torch.einsum(
            "bnd,kd->bkn",
            patches,
            self.direction_queries.to(dtype=patches.dtype, device=patches.device),
        ) / patches.shape[-1] ** 0.5
        gates = torch.sigmoid(self.token_gate(patches)).transpose(1, 2)
        weights = (direction_scores.softmax(dim=-1) * gates).clamp_min(1e-6)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        direction_summaries = torch.einsum("bkn,bnd->bkd", weights, patches)
        direction_weights = torch.einsum(
            "bkd,d->bk",
            direction_summaries,
            self.path_query.to(dtype=patches.dtype, device=patches.device),
        ).softmax(dim=-1)
        kept = torch.einsum("bk,bkd->bd", direction_weights, direction_summaries)
        return kept, direction_summaries.mean(dim=1)

    def slot_gated_pool(self, patches: torch.Tensor):
        slots = self.path_slots.to(dtype=patches.dtype, device=patches.device)
        slots = slots.unsqueeze(0).expand(patches.shape[0], -1, -1)
        gates = torch.sigmoid(self.token_gate(patches)).squeeze(-1).clamp_min(1e-4)
        for _ in range(2):
            scores = torch.einsum("bnd,bkd->bkn", patches, slots) / patches.shape[-1] ** 0.5
            weights = (scores.softmax(dim=-1) * gates.unsqueeze(1)).clamp_min(1e-6)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            contexts = torch.einsum("bkn,bnd->bkd", weights, patches)
            slots = self.slot_update(
                contexts.reshape(-1, patches.shape[-1]).float(),
                slots.reshape(-1, patches.shape[-1]).float(),
            ).to(dtype=patches.dtype).reshape_as(slots)
        slot_scores = torch.einsum("bkd,d->bk", slots, self.path_query.to(dtype=patches.dtype, device=patches.device))
        slot_weights = slot_scores.softmax(dim=-1)
        selected = torch.einsum("bk,bkd->bd", slot_weights, slots)
        return selected, slots.mean(dim=1)

    def multihop_pool(self, patches: torch.Tensor, attention_bias: torch.Tensor = None):
        state = self.path_query.to(dtype=patches.dtype, device=patches.device)
        state = state.unsqueeze(0).expand(patches.shape[0], -1)
        for _ in range(self.hops):
            scores = torch.einsum("bnd,bd->bn", patches, state) / patches.shape[-1] ** 0.5
            if attention_bias is not None:
                scores = scores + attention_bias
            context = torch.einsum("bn,bnd->bd", scores.softmax(dim=1), patches)
            state = self.hop_update(context.float(), state.float()).to(dtype=patches.dtype)
        return state

    def multiquery_multihop_pool(self, patches: torch.Tensor):
        states = self.path_queries.to(dtype=patches.dtype, device=patches.device)
        states = states.unsqueeze(0).expand(patches.shape[0], -1, -1)
        for _ in range(self.hops):
            scores = torch.einsum("bnd,bkd->bkn", patches, states) / patches.shape[-1] ** 0.5
            contexts = torch.einsum("bkn,bnd->bkd", scores.softmax(dim=-1), patches)
            flat_contexts = contexts.reshape(-1, patches.shape[-1])
            flat_states = states.reshape(-1, patches.shape[-1])
            states = self.hop_update(flat_contexts.float(), flat_states.float()).to(dtype=patches.dtype)
            states = states.reshape(patches.shape[0], -1, patches.shape[-1])
        query_scores = torch.einsum("bkd,d->bk", states, self.path_query.to(dtype=patches.dtype, device=patches.device))
        query_weights = query_scores.softmax(dim=-1)
        return torch.einsum("bk,bkd->bd", query_weights, states)

    def forward(self, tokens: torch.Tensor, layerwise_tokens=None):
        cls = tokens[:, 0, :]
        patches = tokens[:, 1:, :]

        if self.mode == "start_pool":
            start, _ = self.attention_pool(patches, self.start_query)
            pooled = torch.cat([cls, start, start], dim=-1)
        elif self.mode == "path_state":
            state_tokens = self.propagate(patches)
            path, _ = self.attention_pool(state_tokens, self.path_query)
            pooled = torch.cat([cls, path, state_tokens.mean(dim=1)], dim=-1)
        elif self.mode == "decoy_aware":
            kept = self.gated_pool(patches)
            suppressed = patches.mean(dim=1) - kept
            pooled = torch.cat([cls, kept, suppressed], dim=-1)
        elif self.mode == "multihead_decoy_aware":
            kept, diversity = self.multihead_gated_pool(patches)
            suppressed = patches.mean(dim=1) - kept
            pooled = torch.cat([cls, kept + diversity, suppressed], dim=-1)
        elif self.mode == "directional_decoy_aware":
            kept, directional = self.directional_gated_pool(patches)
            suppressed = patches.mean(dim=1) - kept
            pooled = torch.cat([cls, kept + directional, suppressed], dim=-1)
        elif self.mode == "adaptive_decoy_aware":
            kept = self.gated_pool(patches)
            suppressed = patches.mean(dim=1) - kept
            alpha = torch.sigmoid(self.router(cls)).to(dtype=cls.dtype)
            pooled = torch.cat([cls, alpha * kept, alpha * suppressed], dim=-1)
        elif self.mode == "adaptive_decoy_aware_mean_supp":
            kept = self.gated_pool(patches)
            suppressed = patches.mean(dim=1)
            alpha = torch.sigmoid(self.router(cls)).to(dtype=cls.dtype)
            pooled = torch.cat([cls, alpha * kept, alpha * suppressed], dim=-1)
        elif self.mode == "adaptive_decoy_aware_no_outer_cls":
            kept = self.gated_pool(patches)
            suppressed = patches.mean(dim=1) - kept
            alpha = torch.sigmoid(self.router(cls)).to(dtype=cls.dtype)
            pooled = torch.cat([cls, alpha * kept, alpha * suppressed], dim=-1)
            return self.residual_scale.to(dtype=cls.dtype) * self.fuse(pooled)
        elif self.mode == "adaptive_decoy_aware_no_inner_cls":
            kept = self.gated_pool(patches)
            suppressed = patches.mean(dim=1) - kept
            alpha = torch.sigmoid(self.router(cls)).to(dtype=cls.dtype)
            pooled = torch.cat([alpha * kept, alpha * suppressed], dim=-1)
            return cls + self.residual_scale.to(dtype=cls.dtype) * self.fuse_two(pooled)
        elif self.mode == "adaptive_decoy_aware_static_alpha":
            kept = self.gated_pool(patches)
            suppressed = patches.mean(dim=1) - kept
            alpha = torch.sigmoid(self.static_alpha).to(dtype=cls.dtype)
            pooled = torch.cat([cls, alpha * kept, alpha * suppressed], dim=-1)
        elif self.mode == "adaptive_decoy_aware_relu_supp":
            kept = self.gated_pool(patches)
            suppressed = torch.relu(patches.mean(dim=1) - kept)
            alpha = torch.sigmoid(self.router(cls)).to(dtype=cls.dtype)
            pooled = torch.cat([cls, alpha * kept, alpha * suppressed], dim=-1)
        elif self.mode == "geo_complement_adar":
            kept, complement = self.complement_gated_pool(patches)
            alpha = torch.sigmoid(self.router(cls)).to(dtype=cls.dtype)
            pooled = torch.cat([cls, alpha * kept, alpha * complement], dim=-1)
        elif self.mode == "geo_twoquery_adar":
            connected, _ = self.attention_pool(patches, self.path_query)
            barrier, _ = self.attention_pool(patches, self.start_query)
            alpha = torch.sigmoid(self.router(cls)).to(dtype=cls.dtype)
            pooled = torch.cat([cls, alpha * connected, alpha * barrier], dim=-1)
        elif self.mode == "geo_band_adar":
            kept, complement = self.spatial_band_pool(patches)
            alpha = torch.sigmoid(self.router(cls)).to(dtype=cls.dtype)
            pooled = torch.cat([cls, alpha * kept, alpha * complement], dim=-1)
        elif self.mode == "adaptive_keep_only":
            kept = self.gated_pool(patches)
            alpha = torch.sigmoid(self.router(cls)).to(dtype=cls.dtype)
            pooled = torch.cat([cls, alpha * kept, torch.zeros_like(kept)], dim=-1)
        elif self.mode == "adaptive_supp_only":
            kept = self.gated_pool(patches)
            suppressed = patches.mean(dim=1) - kept
            alpha = torch.sigmoid(self.router(cls)).to(dtype=cls.dtype)
            pooled = torch.cat([cls, torch.zeros_like(suppressed), alpha * suppressed], dim=-1)
        elif self.mode == "adaptive_keep_only_2branch":
            kept = self.gated_pool(patches)
            alpha = torch.sigmoid(self.router(cls)).to(dtype=cls.dtype)
            pooled = torch.cat([cls, alpha * kept], dim=-1)
            return cls + self.residual_scale.to(dtype=cls.dtype) * self.fuse_two(pooled)
        elif self.mode == "adaptive_residual_decoy_aware":
            kept = self.gated_pool(patches)
            suppressed = patches.mean(dim=1) - kept
            alpha = torch.sigmoid(self.router(cls)).to(dtype=cls.dtype)
            pooled = torch.cat([cls, alpha * kept, alpha * suppressed], dim=-1)
            return cls + self.residual_scale.to(dtype=cls.dtype) * alpha * self.fuse(pooled)
        elif self.mode == "experimental_attention_pool_adar":
            path = self.cls_query_pool(patches, cls)
            suppressed = patches.mean(dim=1) - path
            alpha = torch.sigmoid(self.router(cls)).to(dtype=cls.dtype)
            pooled = torch.cat([cls, alpha * path, alpha * suppressed], dim=-1)
        elif self.mode == "experimental_decoy_contrast_adar":
            kept, decoy = self.decoy_contrast_pool(patches)
            alpha = torch.sigmoid(self.router(cls)).to(dtype=cls.dtype)
            pooled = torch.cat([cls, alpha * kept, alpha * (kept - decoy)], dim=-1)
        elif self.mode == "experimental_multiquery_adar":
            state = self.multiquery_multihop_pool(patches)
            kept, decoy = self.decoy_contrast_pool(patches)
            alpha = torch.sigmoid(self.router(cls)).to(dtype=cls.dtype)
            pooled = torch.cat([cls, alpha * (state + kept), alpha * decoy], dim=-1)
        elif self.mode == "slot_decoy_aware":
            selected, slot_mean = self.slot_gated_pool(patches)
            suppressed = patches.mean(dim=1) - selected
            pooled = torch.cat([cls, selected + slot_mean, suppressed], dim=-1)
        elif self.mode == "adaptive_slot_decoy_aware":
            selected, slot_mean = self.slot_gated_pool(patches)
            suppressed = patches.mean(dim=1) - selected
            alpha = torch.sigmoid(self.router(cls)).to(dtype=cls.dtype)
            pooled = torch.cat([cls, alpha * (selected + slot_mean), alpha * suppressed], dim=-1)
        elif self.mode == "layerwise_decoy_aware":
            if layerwise_tokens is None:
                layerwise_tokens = [tokens]
            weights = self.layer_weights[: len(layerwise_tokens)].softmax(dim=0).to(dtype=tokens.dtype, device=tokens.device)
            mixed = sum(weight * state for weight, state in zip(weights, layerwise_tokens))
            patches = mixed[:, 1:, :]
            kept = self.gated_pool(patches)
            suppressed = patches.mean(dim=1) - kept
            pooled = torch.cat([cls, kept, suppressed], dim=-1)
        elif self.mode == "geo_layer_mixed_adar":
            if layerwise_tokens is None:
                layerwise_tokens = [tokens]
            weights = self.layer_weights[: len(layerwise_tokens)].softmax(dim=0).to(dtype=tokens.dtype, device=tokens.device)
            mixed = sum(weight * state for weight, state in zip(weights, layerwise_tokens))
            patches = mixed[:, 1:, :]
            kept, complement = self.complement_gated_pool(patches)
            alpha = torch.sigmoid(self.router(cls)).to(dtype=cls.dtype)
            pooled = torch.cat([cls, alpha * kept, alpha * complement], dim=-1)
        elif self.mode == "start_path_state":
            start, start_weights = self.attention_pool(patches, self.start_query)
            state_tokens = self.propagate(patches)
            state = torch.einsum("bn,bnd->bd", start_weights, state_tokens)
            pooled = torch.cat([cls, start, state], dim=-1)
        elif self.mode in {"multihop", "multihop_zero"}:
            state = self.multihop_pool(patches)
            start, _ = self.attention_pool(patches, self.start_query)
            pooled = torch.cat([cls, start, state], dim=-1)
        elif self.mode == "multiquery_multihop":
            state = self.multiquery_multihop_pool(patches)
            start, _ = self.attention_pool(patches, self.start_query)
            pooled = torch.cat([cls, start, state], dim=-1)
        elif self.mode == "decoy_fusion_multihop":
            state = self.multihop_pool(patches)
            kept = self.gated_pool(patches)
            pooled = torch.cat([cls, kept, state], dim=-1)
        elif self.mode == "decoy_multihop":
            gates = torch.sigmoid(self.token_gate(patches)).squeeze(-1).clamp_min(1e-4)
            attention_bias = gates.log()
            state = self.multihop_pool(patches, attention_bias=attention_bias)
            kept = torch.einsum("bn,bnd->bd", gates / gates.sum(dim=1, keepdim=True), patches)
            pooled = torch.cat([cls, kept, state], dim=-1)
        else:
            return cls

        return cls + self.residual_scale.to(dtype=cls.dtype) * self.fuse(pooled)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        vit_path_adapter: str = "none",
        vit_path_adapter_bottleneck: int = 128,
        vit_path_adapter_dropout: float = 0.1,
        vit_path_adapter_hops: int = 4,
        vit_path_adapter_residual_scale: float = 1.0,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)
        if vit_path_adapter == "none":
            self.path_readout = None
        elif vit_path_adapter == "adaptive_supp_only_clean":
            self.path_readout = ViTSuppOnlyReadout(
                width=width,
                bottleneck=vit_path_adapter_bottleneck,
                dropout=vit_path_adapter_dropout,
                residual_scale=vit_path_adapter_residual_scale,
            )
        elif vit_path_adapter == "adaptive_keep_only_clean":
            self.path_readout = ViTKeepOnlyReadout(
                width=width,
                bottleneck=vit_path_adapter_bottleneck,
                dropout=vit_path_adapter_dropout,
                residual_scale=vit_path_adapter_residual_scale,
            )
        else:
            self.path_readout = ViTPathReadout(
                width=width,
                grid_size=input_resolution // patch_size,
                mode=vit_path_adapter,
                bottleneck=vit_path_adapter_bottleneck,
                dropout=vit_path_adapter_dropout,
                hops=vit_path_adapter_hops,
                residual_scale=vit_path_adapter_residual_scale,
            )

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        layerwise_tokens = None
        if self.path_readout is not None and getattr(self.path_readout, "wants_layerwise_tokens", False):
            states = []
            for block in self.transformer.resblocks:
                x = block(x)
                states.append(x.permute(1, 0, 2))
            layerwise_tokens = states[-4:]
            x = layerwise_tokens[-1]
        else:
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD

        if self.path_readout is not None:
            x = self.path_readout(x, layerwise_tokens=layerwise_tokens)
        else:
            x = x[:, 0, :]
        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # new
                 vision_backbone: str = "vit",
                 vision_model_name: str = "vim_base",
                 vision_ckpt_path: str = None,
                 path_adapter_layers: int = 0,
                 path_adapter_bottleneck: int = 128,
                 path_adapter_dropout: float = 0.1,
                 path_adapter_pooling: str = "forward",
                 vit_path_adapter: str = "none",
                 vit_path_adapter_bottleneck: int = 128,
                 vit_path_adapter_dropout: float = 0.1,
                 vit_path_adapter_hops: int = 4,
                 vit_path_adapter_residual_scale: float = 1.0,
                 ):
        super().__init__()

        self.context_length = context_length
        self.vision_backbone = vision_backbone

        if vision_backbone == "vim":
            from .vim_wrapper import VimVisionWrapper
            # 先固定用 Vim Base，輸出維度對齊 embed_dim
            self.visual = VimVisionWrapper(
                model_name=vision_model_name,
                output_dim=embed_dim,
                pretrained=False,
                ckpt_path=vision_ckpt_path,
                path_adapter_layers=path_adapter_layers,
                path_adapter_bottleneck=path_adapter_bottleneck,
                path_adapter_dropout=path_adapter_dropout,
                path_adapter_pooling=path_adapter_pooling,
            )

        elif isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                vit_path_adapter=vit_path_adapter,
                vit_path_adapter_bottleneck=vit_path_adapter_bottleneck,
                vit_path_adapter_dropout=vit_path_adapter_dropout,
                vit_path_adapter_hops=vit_path_adapter_hops,
                vit_path_adapter_residual_scale=vit_path_adapter_residual_scale,
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        # ===== Vim backbone =====
        # 如果是 VimVisionWrapper，就不要重設 image backbone 權重
        elif getattr(self, "vision_backbone", "vit") == "vim":

            # 初始化 projection（支援 Linear / Sequential）
            if hasattr(self.visual, "proj") and self.visual.proj is not None:

                if isinstance(self.visual.proj, nn.Linear):
                    nn.init.normal_(self.visual.proj.weight, std=self.transformer.width ** -0.5)
                    if self.visual.proj.bias is not None:
                        nn.init.zeros_(self.visual.proj.bias)

                elif isinstance(self.visual.proj, nn.Sequential):
                    for m in self.visual.proj.modules():
                        if isinstance(m, nn.Linear):
                            nn.init.normal_(m.weight, std=self.transformer.width ** -0.5)
                            if m.bias is not None:
                                nn.init.zeros_(m.bias)

            if hasattr(self.visual, "ln_post"):
                if hasattr(self.visual.ln_post, "weight") and self.visual.ln_post.weight is not None:
                    nn.init.ones_(self.visual.ln_post.weight)
                if hasattr(self.visual.ln_post, "bias") and self.visual.ln_post.bias is not None:
                    nn.init.zeros_(self.visual.ln_post.bias)

        # ===== Text transformer =====
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        # 修改成更泛化的版本能支援 Vim
        return next(self.parameters()).dtype
        # return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if isinstance(attr, torch.Tensor):
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(
    state_dict: dict,
    vision_backbone: str = "vit",
    vision_model_name: str = "vim_base",
    vision_ckpt_path: str = None,
    path_adapter_layers: int = 0,
    path_adapter_bottleneck: int = 128,
    path_adapter_dropout: float = 0.1,
    path_adapter_pooling: str = "forward",
    vit_path_adapter: str = "none",
    vit_path_adapter_bottleneck: int = 128,
    vit_path_adapter_dropout: float = 0.1,
    vit_path_adapter_hops: int = 4,
    vit_path_adapter_residual_scale: float = 1.0,
):
    vit = "visual.proj" in state_dict

    # for k, v in state_dict.items():
    #     if "visual" in k:
    #         print(k, v.shape)

    if vision_backbone == "vim":
        vision_width = 768
        vision_layers = 24   # 這裡只是佔位，不會真的用到原本 VisionTransformer
        vision_patch_size = 16
        image_resolution = 224
    else:
        # 你原本的 vit / resnet 判斷邏輯
        if vit:
            vision_width = state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        vision_backbone=vision_backbone,
        vision_model_name=vision_model_name,
        vision_ckpt_path=vision_ckpt_path,
        path_adapter_layers=path_adapter_layers,
        path_adapter_bottleneck=path_adapter_bottleneck,
        path_adapter_dropout=path_adapter_dropout,
        path_adapter_pooling=path_adapter_pooling,
        vit_path_adapter=vit_path_adapter,
        vit_path_adapter_bottleneck=vit_path_adapter_bottleneck,
        vit_path_adapter_dropout=vit_path_adapter_dropout,
        vit_path_adapter_hops=vit_path_adapter_hops,
        vit_path_adapter_residual_scale=vit_path_adapter_residual_scale,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    if vision_backbone != "vim":
        convert_weights(model)

    if vision_backbone == "vim":
        print("Loading CLIP weights (excluding vit)...")

        # 過濾掉 visual.*
        new_state_dict = {
            k: v for k, v in state_dict.items()
            if not k.startswith("visual.")
        }

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

        print("Unexpected keys:", unexpected)
    else:
        if vit_path_adapter == "none":
            model.load_state_dict(state_dict)
        else:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print("Loading CLIP weights with ViT path adapter...")
            print("Missing keys:", [key for key in missing if key.startswith("visual.path_readout")])
            print("Unexpected keys:", unexpected)

    return model.eval()
