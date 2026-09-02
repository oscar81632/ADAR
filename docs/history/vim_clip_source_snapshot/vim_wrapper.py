import torch
import torch.nn as nn
import sys
# 這裡要 import Vim 位置
sys.path.append("<LOCAL_ROOT>/Vim1")

# 這個 import 路徑要依你的實際 Vim repo 位置調整
from vim.models_mamba import (
    RMSNorm,
    layer_norm_fn,
    rms_norm_fn,
    vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2,
)

class ResidualProjectionHead(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        bottleneck_dim,
        output_dim,
        dropout1=0.35,
        dropout2=0.55,
        skip_dropout=0.6,
    ):
        super().__init__()
        self.skip = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(skip_dropout),
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(bottleneck_dim, output_dim),
        )
        self.out_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        return self.out_norm(self.skip(x) + self.mlp(x))


class PathAwareStateAdapter(nn.Module):
    def __init__(self, dim, bottleneck_dim=128, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, dim),
            nn.Sigmoid(),
        )

        nn.init.normal_(self.up.weight, std=1e-4)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states):
        context = hidden_states.mean(dim=1, keepdim=True).expand_as(hidden_states)
        update = self.up(self.dropout(self.act(self.down(hidden_states))))
        gate = self.gate(torch.cat([hidden_states, context], dim=-1))
        return hidden_states + gate * update


class PathAwareTokenPooler(nn.Module):
    def __init__(self, dim, bottleneck_dim=128, dropout=0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states, cls_index):
        cls_state = hidden_states[:, cls_index, :]
        patch_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        patch_mask[:, cls_index] = False
        patch_states = hidden_states[patch_mask].view(hidden_states.shape[0], hidden_states.shape[1] - 1, hidden_states.shape[2])
        context = patch_states.mean(dim=1, keepdim=True).expand_as(patch_states)
        gate_logits = self.gate(torch.cat([patch_states, context], dim=-1)).squeeze(-1)
        weights = torch.softmax(gate_logits, dim=1).unsqueeze(-1)
        pooled = (weights * patch_states).sum(dim=1)
        return self.out_norm(cls_state + self.mix(torch.cat([cls_state, pooled], dim=-1)))


class BidirectionalPathAwareTokenPooler(nn.Module):
    def __init__(self, dim, bottleneck_dim=128, dropout=0.1):
        super().__init__()
        self.forward_gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.backward_gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def weighted_pool(self, patch_states, gate):
        context = patch_states.mean(dim=1, keepdim=True).expand_as(patch_states)
        gate_logits = gate(torch.cat([patch_states, context], dim=-1)).squeeze(-1)
        weights = torch.softmax(gate_logits, dim=1).unsqueeze(-1)
        return (weights * patch_states).sum(dim=1)

    def forward(self, hidden_states, cls_index):
        cls_state = hidden_states[:, cls_index, :]
        patch_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        patch_mask[:, cls_index] = False
        patch_states = hidden_states[patch_mask].view(hidden_states.shape[0], hidden_states.shape[1] - 1, hidden_states.shape[2])
        forward_pooled = self.weighted_pool(patch_states, self.forward_gate)
        backward_pooled = self.weighted_pool(torch.flip(patch_states, dims=[1]), self.backward_gate)
        mixed = self.mix(torch.cat([cls_state, forward_pooled, backward_pooled], dim=-1))
        return self.out_norm(cls_state + mixed)


class StartAwareBidirectionalPathAwareTokenPooler(nn.Module):
    def __init__(self, dim, bottleneck_dim=128, dropout=0.1):
        super().__init__()
        self.start_query = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.forward_gate = nn.Sequential(
            nn.Linear(dim * 4, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.backward_gate = nn.Sequential(
            nn.Linear(dim * 4, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def get_patch_states(self, hidden_states, cls_index):
        patch_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        patch_mask[:, cls_index] = False
        return hidden_states[patch_mask].view(
            hidden_states.shape[0],
            hidden_states.shape[1] - 1,
            hidden_states.shape[2],
        )

    def infer_start_context(self, patch_states):
        start_logits = self.start_query(patch_states).squeeze(-1)
        start_weights = torch.softmax(start_logits, dim=1).unsqueeze(-1)
        return (start_weights * patch_states).sum(dim=1)

    def weighted_pool(self, patch_states, global_context, start_context, gate):
        global_context = global_context.unsqueeze(1).expand_as(patch_states)
        start_context = start_context.unsqueeze(1).expand_as(patch_states)
        relative_to_start = patch_states - start_context
        gate_input = torch.cat([patch_states, global_context, start_context, relative_to_start], dim=-1)
        gate_logits = gate(gate_input).squeeze(-1)
        weights = torch.softmax(gate_logits, dim=1).unsqueeze(-1)
        return (weights * patch_states).sum(dim=1)

    def forward(self, hidden_states, cls_index):
        cls_state = hidden_states[:, cls_index, :]
        patch_states = self.get_patch_states(hidden_states, cls_index)
        global_context = patch_states.mean(dim=1)
        start_context = self.infer_start_context(patch_states)
        forward_pooled = self.weighted_pool(patch_states, global_context, start_context, self.forward_gate)
        backward_pooled = self.weighted_pool(
            torch.flip(patch_states, dims=[1]),
            global_context,
            start_context,
            self.backward_gate,
        )
        mixed = self.mix(torch.cat([cls_state, start_context, forward_pooled, backward_pooled], dim=-1))
        return self.out_norm(cls_state + mixed)


class GeometryAwareBidirectionalPathAwareTokenPooler(nn.Module):
    def __init__(self, dim, bottleneck_dim=128, dropout=0.1):
        super().__init__()
        self.forward_gate = nn.Sequential(
            nn.Linear(dim * 4, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.backward_gate = nn.Sequential(
            nn.Linear(dim * 4, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.column_down_gate = nn.Sequential(
            nn.Linear(dim * 4, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.column_up_gate = nn.Sequential(
            nn.Linear(dim * 4, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(dim * 5),
            nn.Linear(dim * 5, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def get_patch_states(self, hidden_states, cls_index):
        patch_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        patch_mask[:, cls_index] = False
        return hidden_states[patch_mask].view(
            hidden_states.shape[0],
            hidden_states.shape[1] - 1,
            hidden_states.shape[2],
        )

    def add_geometry_context(self, patch_states):
        bsz, num_patches, dim = patch_states.shape
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise RuntimeError(f"Expected square patch grid, got {num_patches} patches")
        grid = patch_states.view(bsz, grid_size, grid_size, dim)
        row_context = grid.mean(dim=2, keepdim=True).expand_as(grid).reshape(bsz, num_patches, dim)
        col_context = grid.mean(dim=1, keepdim=True).expand_as(grid).reshape(bsz, num_patches, dim)
        global_context = patch_states.mean(dim=1, keepdim=True).expand_as(patch_states)
        return row_context, col_context, global_context, grid_size

    def weighted_pool(self, patch_states, row_context, col_context, global_context, gate):
        gate_input = torch.cat([patch_states, row_context, col_context, global_context], dim=-1)
        gate_logits = gate(gate_input).squeeze(-1)
        weights = torch.softmax(gate_logits, dim=1).unsqueeze(-1)
        return (weights * patch_states).sum(dim=1)

    def forward(self, hidden_states, cls_index):
        cls_state = hidden_states[:, cls_index, :]
        patch_states = self.get_patch_states(hidden_states, cls_index)
        row_context, col_context, global_context, grid_size = self.add_geometry_context(patch_states)

        forward_pooled = self.weighted_pool(
            patch_states, row_context, col_context, global_context, self.forward_gate
        )
        backward_pooled = self.weighted_pool(
            torch.flip(patch_states, dims=[1]),
            torch.flip(row_context, dims=[1]),
            torch.flip(col_context, dims=[1]),
            torch.flip(global_context, dims=[1]),
            self.backward_gate,
        )

        column_order = torch.arange(patch_states.shape[1], device=patch_states.device).view(grid_size, grid_size).t().reshape(-1)
        column_states = patch_states[:, column_order, :]
        column_rows = row_context[:, column_order, :]
        column_cols = col_context[:, column_order, :]
        column_global = global_context[:, column_order, :]
        column_down_pooled = self.weighted_pool(
            column_states, column_rows, column_cols, column_global, self.column_down_gate
        )
        column_up_pooled = self.weighted_pool(
            torch.flip(column_states, dims=[1]),
            torch.flip(column_rows, dims=[1]),
            torch.flip(column_cols, dims=[1]),
            torch.flip(column_global, dims=[1]),
            self.column_up_gate,
        )

        mixed = self.mix(
            torch.cat([cls_state, forward_pooled, backward_pooled, column_down_pooled, column_up_pooled], dim=-1)
        )
        return self.out_norm(cls_state + mixed)


class PathStatePropagationTokenPooler(nn.Module):
    def __init__(self, dim, bottleneck_dim=128, dropout=0.1, propagation_steps=3):
        super().__init__()
        self.propagation_steps = propagation_steps
        self.message_norm = nn.LayerNorm(dim * 3)
        self.message = nn.Sequential(
            nn.Linear(dim * 3, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )
        self.message_gate = nn.Sequential(
            nn.Linear(dim * 3, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
            nn.Sigmoid(),
        )
        self.update_norm = nn.LayerNorm(dim)
        self.attn_gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def get_patch_states(self, hidden_states, cls_index):
        patch_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        patch_mask[:, cls_index] = False
        return hidden_states[patch_mask].view(
            hidden_states.shape[0],
            hidden_states.shape[1] - 1,
            hidden_states.shape[2],
        )

    def neighbor_states(self, grid_states):
        bsz, grid_size, _, dim = grid_states.shape
        zeros = torch.zeros(bsz, grid_size, 1, dim, device=grid_states.device, dtype=grid_states.dtype)
        left = torch.cat([zeros, grid_states[:, :, :-1, :]], dim=2)
        right = torch.cat([grid_states[:, :, 1:, :], zeros], dim=2)

        zeros = torch.zeros(bsz, 1, grid_size, dim, device=grid_states.device, dtype=grid_states.dtype)
        up = torch.cat([zeros, grid_states[:, :-1, :, :]], dim=1)
        down = torch.cat([grid_states[:, 1:, :, :], zeros], dim=1)
        return [left, right, up, down]

    def propagate_once(self, grid_states, global_context):
        global_context = global_context[:, None, None, :].expand_as(grid_states)
        updates = []
        for neighbor in self.neighbor_states(grid_states):
            message_input = torch.cat([grid_states, neighbor, global_context], dim=-1)
            message_input = self.message_norm(message_input)
            updates.append(self.message_gate(message_input) * self.message(message_input))
        update = torch.stack(updates, dim=0).mean(dim=0)
        return self.update_norm(grid_states + update)

    def propagate(self, patch_states):
        bsz, num_patches, dim = patch_states.shape
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise RuntimeError(f"Expected square patch grid, got {num_patches} patches")
        grid_states = patch_states.view(bsz, grid_size, grid_size, dim)
        global_context = patch_states.mean(dim=1)
        for _ in range(self.propagation_steps):
            grid_states = self.propagate_once(grid_states, global_context)
        return grid_states.reshape(bsz, num_patches, dim)

    def weighted_pool(self, states, context):
        context = context.unsqueeze(1).expand_as(states)
        logits = self.attn_gate(torch.cat([states, context], dim=-1)).squeeze(-1)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (weights * states).sum(dim=1)

    def forward(self, hidden_states, cls_index):
        cls_state = hidden_states[:, cls_index, :]
        patch_states = self.get_patch_states(hidden_states, cls_index)
        propagated_states = self.propagate(patch_states)
        context = propagated_states.mean(dim=1)
        original_pooled = self.weighted_pool(patch_states, context)
        propagated_pooled = self.weighted_pool(propagated_states, context)
        mixed = self.mix(torch.cat([cls_state, original_pooled, propagated_pooled], dim=-1))
        return self.out_norm(cls_state + mixed)


class HybridPathStateBidirectionalTokenPooler(nn.Module):
    def __init__(self, dim, bottleneck_dim=128, dropout=0.1, propagation_steps=3):
        super().__init__()
        self.propagation_steps = propagation_steps
        self.forward_gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.backward_gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.message_norm = nn.LayerNorm(dim * 3)
        self.message = nn.Sequential(
            nn.Linear(dim * 3, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )
        self.message_gate = nn.Sequential(
            nn.Linear(dim * 3, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
            nn.Sigmoid(),
        )
        self.update_norm = nn.LayerNorm(dim)
        self.propagated_gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def get_patch_states(self, hidden_states, cls_index):
        patch_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        patch_mask[:, cls_index] = False
        return hidden_states[patch_mask].view(
            hidden_states.shape[0],
            hidden_states.shape[1] - 1,
            hidden_states.shape[2],
        )

    def weighted_pool(self, states, gate):
        context = states.mean(dim=1, keepdim=True).expand_as(states)
        logits = gate(torch.cat([states, context], dim=-1)).squeeze(-1)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (weights * states).sum(dim=1)

    def neighbor_states(self, grid_states):
        bsz, grid_size, _, dim = grid_states.shape
        zeros = torch.zeros(bsz, grid_size, 1, dim, device=grid_states.device, dtype=grid_states.dtype)
        left = torch.cat([zeros, grid_states[:, :, :-1, :]], dim=2)
        right = torch.cat([grid_states[:, :, 1:, :], zeros], dim=2)

        zeros = torch.zeros(bsz, 1, grid_size, dim, device=grid_states.device, dtype=grid_states.dtype)
        up = torch.cat([zeros, grid_states[:, :-1, :, :]], dim=1)
        down = torch.cat([grid_states[:, 1:, :, :], zeros], dim=1)
        return [left, right, up, down]

    def propagate_once(self, grid_states, global_context):
        global_context = global_context[:, None, None, :].expand_as(grid_states)
        updates = []
        for neighbor in self.neighbor_states(grid_states):
            message_input = torch.cat([grid_states, neighbor, global_context], dim=-1)
            message_input = self.message_norm(message_input)
            updates.append(self.message_gate(message_input) * self.message(message_input))
        update = torch.stack(updates, dim=0).mean(dim=0)
        return self.update_norm(grid_states + update)

    def propagate(self, patch_states):
        bsz, num_patches, dim = patch_states.shape
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise RuntimeError(f"Expected square patch grid, got {num_patches} patches")
        grid_states = patch_states.view(bsz, grid_size, grid_size, dim)
        global_context = patch_states.mean(dim=1)
        for _ in range(self.propagation_steps):
            grid_states = self.propagate_once(grid_states, global_context)
        return grid_states.reshape(bsz, num_patches, dim)

    def forward(self, hidden_states, cls_index):
        cls_state = hidden_states[:, cls_index, :]
        patch_states = self.get_patch_states(hidden_states, cls_index)
        forward_pooled = self.weighted_pool(patch_states, self.forward_gate)
        backward_pooled = self.weighted_pool(torch.flip(patch_states, dims=[1]), self.backward_gate)
        propagated_states = self.propagate(patch_states)
        propagated_pooled = self.weighted_pool(propagated_states, self.propagated_gate)
        mixed = self.mix(torch.cat([cls_state, forward_pooled, backward_pooled, propagated_pooled], dim=-1))
        return self.out_norm(cls_state + mixed)


class GatedHybridPathStateTokenPooler(nn.Module):
    def __init__(self, dim, bottleneck_dim=128, dropout=0.1, propagation_steps=3):
        super().__init__()
        self.propagation_steps = propagation_steps
        self.forward_gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.backward_gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.original_gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.propagated_gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.message_norm = nn.LayerNorm(dim * 3)
        self.message = nn.Sequential(
            nn.Linear(dim * 3, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )
        self.message_gate = nn.Sequential(
            nn.Linear(dim * 3, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
            nn.Sigmoid(),
        )
        self.update_norm = nn.LayerNorm(dim)
        self.expert_gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 4),
        )
        self.expert_proj = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
                for _ in range(4)
            ]
        )
        self.out_norm = nn.LayerNorm(dim)

    def get_patch_states(self, hidden_states, cls_index):
        patch_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        patch_mask[:, cls_index] = False
        return hidden_states[patch_mask].view(
            hidden_states.shape[0],
            hidden_states.shape[1] - 1,
            hidden_states.shape[2],
        )

    def weighted_pool(self, states, gate):
        context = states.mean(dim=1, keepdim=True).expand_as(states)
        logits = gate(torch.cat([states, context], dim=-1)).squeeze(-1)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (weights * states).sum(dim=1)

    def neighbor_states(self, grid_states):
        bsz, grid_size, _, dim = grid_states.shape
        zeros = torch.zeros(bsz, grid_size, 1, dim, device=grid_states.device, dtype=grid_states.dtype)
        left = torch.cat([zeros, grid_states[:, :, :-1, :]], dim=2)
        right = torch.cat([grid_states[:, :, 1:, :], zeros], dim=2)

        zeros = torch.zeros(bsz, 1, grid_size, dim, device=grid_states.device, dtype=grid_states.dtype)
        up = torch.cat([zeros, grid_states[:, :-1, :, :]], dim=1)
        down = torch.cat([grid_states[:, 1:, :, :], zeros], dim=1)
        return [left, right, up, down]

    def propagate_once(self, grid_states, global_context):
        global_context = global_context[:, None, None, :].expand_as(grid_states)
        updates = []
        for neighbor in self.neighbor_states(grid_states):
            message_input = torch.cat([grid_states, neighbor, global_context], dim=-1)
            message_input = self.message_norm(message_input)
            updates.append(self.message_gate(message_input) * self.message(message_input))
        update = torch.stack(updates, dim=0).mean(dim=0)
        return self.update_norm(grid_states + update)

    def propagate(self, patch_states):
        bsz, num_patches, dim = patch_states.shape
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise RuntimeError(f"Expected square patch grid, got {num_patches} patches")
        grid_states = patch_states.view(bsz, grid_size, grid_size, dim)
        global_context = patch_states.mean(dim=1)
        for _ in range(self.propagation_steps):
            grid_states = self.propagate_once(grid_states, global_context)
        return grid_states.reshape(bsz, num_patches, dim)

    def forward(self, hidden_states, cls_index):
        cls_state = hidden_states[:, cls_index, :]
        patch_states = self.get_patch_states(hidden_states, cls_index)
        propagated_states = self.propagate(patch_states)

        experts = torch.stack(
            [
                self.weighted_pool(patch_states, self.original_gate),
                self.weighted_pool(patch_states, self.forward_gate),
                self.weighted_pool(torch.flip(patch_states, dims=[1]), self.backward_gate),
                self.weighted_pool(propagated_states, self.propagated_gate),
            ],
            dim=1,
        )
        projected_experts = torch.stack(
            [proj(experts[:, idx, :]) for idx, proj in enumerate(self.expert_proj)],
            dim=1,
        )
        global_context = patch_states.mean(dim=1)
        expert_weights = torch.softmax(self.expert_gate(torch.cat([cls_state, global_context], dim=-1)), dim=-1)
        mixed = (expert_weights.unsqueeze(-1) * projected_experts).sum(dim=1)
        return self.out_norm(cls_state + mixed)


class OmniPathStateTokenPooler(nn.Module):
    def __init__(self, dim, bottleneck_dim=128, dropout=0.1, propagation_steps=3):
        super().__init__()
        self.propagation_steps = propagation_steps
        self.row_forward_gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.row_backward_gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.col_down_gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.col_up_gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.propagated_gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.message_norm = nn.LayerNorm(dim * 3)
        self.message = nn.Sequential(
            nn.Linear(dim * 3, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )
        self.message_gate = nn.Sequential(
            nn.Linear(dim * 3, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
            nn.Sigmoid(),
        )
        self.update_norm = nn.LayerNorm(dim)
        self.mix = nn.Sequential(
            nn.LayerNorm(dim * 6),
            nn.Linear(dim * 6, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def get_patch_states(self, hidden_states, cls_index):
        patch_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        patch_mask[:, cls_index] = False
        return hidden_states[patch_mask].view(
            hidden_states.shape[0],
            hidden_states.shape[1] - 1,
            hidden_states.shape[2],
        )

    def weighted_pool(self, states, gate):
        context = states.mean(dim=1, keepdim=True).expand_as(states)
        logits = gate(torch.cat([states, context], dim=-1)).squeeze(-1)
        weights = torch.softmax(logits, dim=1).unsqueeze(-1)
        return (weights * states).sum(dim=1)

    def column_order(self, patch_states):
        num_patches = patch_states.shape[1]
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise RuntimeError(f"Expected square patch grid, got {num_patches} patches")
        order = torch.arange(num_patches, device=patch_states.device).view(grid_size, grid_size).t().reshape(-1)
        return patch_states[:, order, :]

    def neighbor_states(self, grid_states):
        bsz, grid_size, _, dim = grid_states.shape
        zeros = torch.zeros(bsz, grid_size, 1, dim, device=grid_states.device, dtype=grid_states.dtype)
        left = torch.cat([zeros, grid_states[:, :, :-1, :]], dim=2)
        right = torch.cat([grid_states[:, :, 1:, :], zeros], dim=2)

        zeros = torch.zeros(bsz, 1, grid_size, dim, device=grid_states.device, dtype=grid_states.dtype)
        up = torch.cat([zeros, grid_states[:, :-1, :, :]], dim=1)
        down = torch.cat([grid_states[:, 1:, :, :], zeros], dim=1)
        return [left, right, up, down]

    def propagate_once(self, grid_states, global_context):
        global_context = global_context[:, None, None, :].expand_as(grid_states)
        updates = []
        for neighbor in self.neighbor_states(grid_states):
            message_input = torch.cat([grid_states, neighbor, global_context], dim=-1)
            message_input = self.message_norm(message_input)
            updates.append(self.message_gate(message_input) * self.message(message_input))
        update = torch.stack(updates, dim=0).mean(dim=0)
        return self.update_norm(grid_states + update)

    def propagate(self, patch_states):
        bsz, num_patches, dim = patch_states.shape
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise RuntimeError(f"Expected square patch grid, got {num_patches} patches")
        grid_states = patch_states.view(bsz, grid_size, grid_size, dim)
        global_context = patch_states.mean(dim=1)
        for _ in range(self.propagation_steps):
            grid_states = self.propagate_once(grid_states, global_context)
        return grid_states.reshape(bsz, num_patches, dim)

    def forward(self, hidden_states, cls_index):
        cls_state = hidden_states[:, cls_index, :]
        patch_states = self.get_patch_states(hidden_states, cls_index)
        col_states = self.column_order(patch_states)
        propagated_states = self.propagate(patch_states)
        row_forward = self.weighted_pool(patch_states, self.row_forward_gate)
        row_backward = self.weighted_pool(torch.flip(patch_states, dims=[1]), self.row_backward_gate)
        col_down = self.weighted_pool(col_states, self.col_down_gate)
        col_up = self.weighted_pool(torch.flip(col_states, dims=[1]), self.col_up_gate)
        propagated = self.weighted_pool(propagated_states, self.propagated_gate)
        mixed = self.mix(torch.cat([cls_state, row_forward, row_backward, col_down, col_up, propagated], dim=-1))
        return self.out_norm(cls_state + mixed)


class MemoryOmniPathStateTokenPooler(OmniPathStateTokenPooler):
    def __init__(self, dim, bottleneck_dim=128, dropout=0.1, propagation_steps=3, memory_tokens=2):
        super().__init__(
            dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            propagation_steps=propagation_steps,
        )
        self.memory_tokens = nn.Parameter(torch.zeros(memory_tokens, dim))
        nn.init.normal_(self.memory_tokens, std=0.02)
        self.memory_gate = nn.Sequential(
            nn.Linear(dim * 3, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(dim * 7),
            nn.Linear(dim * 7, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def memory_pool(self, states, cls_state):
        bsz, num_tokens, dim = states.shape
        memories = self.memory_tokens.unsqueeze(0).expand(bsz, -1, -1)
        global_context = states.mean(dim=1, keepdim=True)
        pooled = []
        for idx in range(memories.shape[1]):
            memory = memories[:, idx : idx + 1, :] + cls_state.unsqueeze(1)
            memory = memory.expand(-1, num_tokens, -1)
            context = global_context.expand(-1, num_tokens, -1)
            logits = self.memory_gate(torch.cat([states, memory, context], dim=-1)).squeeze(-1)
            weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
            pooled.append((weights * states).sum(dim=1))
        return torch.stack(pooled, dim=1).mean(dim=1)

    def forward(self, hidden_states, cls_index):
        cls_state = hidden_states[:, cls_index, :]
        patch_states = self.get_patch_states(hidden_states, cls_index)
        col_states = self.column_order(patch_states)
        propagated_states = self.propagate(patch_states)
        row_forward = self.weighted_pool(patch_states, self.row_forward_gate)
        row_backward = self.weighted_pool(torch.flip(patch_states, dims=[1]), self.row_backward_gate)
        col_down = self.weighted_pool(col_states, self.col_down_gate)
        col_up = self.weighted_pool(torch.flip(col_states, dims=[1]), self.col_up_gate)
        propagated = self.weighted_pool(propagated_states, self.propagated_gate)
        memory = self.memory_pool(propagated_states, cls_state)
        mixed = self.mix(
            torch.cat([cls_state, row_forward, row_backward, col_down, col_up, propagated, memory], dim=-1)
        )
        return self.out_norm(cls_state + mixed)


class EndpointRefinementOmniTokenPooler(OmniPathStateTokenPooler):
    def __init__(self, dim, bottleneck_dim=128, dropout=0.1, propagation_steps=3):
        super().__init__(
            dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            propagation_steps=propagation_steps,
        )
        self.endpoint_gate = nn.Sequential(
            nn.Linear(dim * 4, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.endpoint_refine = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(dim * 7),
            nn.Linear(dim * 7, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def endpoint_pool(self, patch_states, propagated_states, path_summary):
        path_context = path_summary.unsqueeze(1).expand_as(patch_states)
        global_context = patch_states.mean(dim=1, keepdim=True).expand_as(patch_states)
        propagated_context = propagated_states.mean(dim=1, keepdim=True).expand_as(patch_states)
        gate_input = torch.cat([patch_states, path_context, global_context, propagated_context], dim=-1)
        logits = self.endpoint_gate(gate_input).squeeze(-1)
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        endpoint = (weights * patch_states).sum(dim=1)
        return self.endpoint_refine(torch.cat([endpoint, path_summary, propagated_states.mean(dim=1)], dim=-1))

    def forward(self, hidden_states, cls_index):
        cls_state = hidden_states[:, cls_index, :]
        patch_states = self.get_patch_states(hidden_states, cls_index)
        col_states = self.column_order(patch_states)
        propagated_states = self.propagate(patch_states)
        row_forward = self.weighted_pool(patch_states, self.row_forward_gate)
        row_backward = self.weighted_pool(torch.flip(patch_states, dims=[1]), self.row_backward_gate)
        col_down = self.weighted_pool(col_states, self.col_down_gate)
        col_up = self.weighted_pool(torch.flip(col_states, dims=[1]), self.col_up_gate)
        propagated = self.weighted_pool(propagated_states, self.propagated_gate)
        path_summary = (row_forward + row_backward + col_down + col_up + propagated) / 5.0
        endpoint = self.endpoint_pool(patch_states, propagated_states, path_summary)
        mixed = self.mix(
            torch.cat([cls_state, row_forward, row_backward, col_down, col_up, propagated, endpoint], dim=-1)
        )
        return self.out_norm(cls_state + mixed)


class LayerwiseOmniPathStateTokenPooler(nn.Module):
    wants_layerwise_states = True

    def __init__(self, dim, bottleneck_dim=128, dropout=0.1, propagation_steps=3):
        super().__init__()
        self.state_norm = nn.LayerNorm(dim)
        self.omni = OmniPathStateTokenPooler(
            dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            propagation_steps=propagation_steps,
        )
        self.summary_gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, hidden_states, cls_index, layerwise_states=None):
        final_summary = self.omni(hidden_states, cls_index)
        if not layerwise_states:
            return final_summary

        summaries = []
        for states in layerwise_states:
            summaries.append(self.omni(self.state_norm(states), cls_index))
        summaries.append(final_summary)
        summaries = torch.stack(summaries, dim=1)

        final_context = final_summary.unsqueeze(1).expand_as(summaries)
        logits = self.summary_gate(torch.cat([summaries, final_context], dim=-1)).squeeze(-1)
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        layer_summary = (weights * summaries).sum(dim=1)
        fused = self.fuse(torch.cat([final_summary, layer_summary], dim=-1))
        return self.out_norm(final_summary + fused)


class DecoySuppressionTokenPooler(nn.Module):
    def __init__(self, dim, bottleneck_dim=128, dropout=0.1, num_hypotheses=4, propagation_steps=3):
        super().__init__()
        self.num_hypotheses = num_hypotheses
        self.propagation_steps = propagation_steps
        self.hypothesis_gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, num_hypotheses),
        )
        self.propagated_hypothesis_gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, num_hypotheses),
        )
        self.message_norm = nn.LayerNorm(dim * 3)
        self.message = nn.Sequential(
            nn.Linear(dim * 3, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )
        self.message_gate = nn.Sequential(
            nn.Linear(dim * 3, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
            nn.Sigmoid(),
        )
        self.update_norm = nn.LayerNorm(dim)
        self.confidence = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.keep_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.suppress_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.suppress_scale = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(dim)

    def get_patch_states(self, hidden_states, cls_index):
        patch_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        patch_mask[:, cls_index] = False
        return hidden_states[patch_mask].view(
            hidden_states.shape[0],
            hidden_states.shape[1] - 1,
            hidden_states.shape[2],
        )

    def neighbor_states(self, grid_states):
        bsz, grid_size, _, dim = grid_states.shape
        zeros = torch.zeros(bsz, grid_size, 1, dim, device=grid_states.device, dtype=grid_states.dtype)
        left = torch.cat([zeros, grid_states[:, :, :-1, :]], dim=2)
        right = torch.cat([grid_states[:, :, 1:, :], zeros], dim=2)

        zeros = torch.zeros(bsz, 1, grid_size, dim, device=grid_states.device, dtype=grid_states.dtype)
        up = torch.cat([zeros, grid_states[:, :-1, :, :]], dim=1)
        down = torch.cat([grid_states[:, 1:, :, :], zeros], dim=1)
        return [left, right, up, down]

    def propagate_once(self, grid_states, global_context):
        global_context = global_context[:, None, None, :].expand_as(grid_states)
        updates = []
        for neighbor in self.neighbor_states(grid_states):
            message_input = torch.cat([grid_states, neighbor, global_context], dim=-1)
            message_input = self.message_norm(message_input)
            updates.append(self.message_gate(message_input) * self.message(message_input))
        update = torch.stack(updates, dim=0).mean(dim=0)
        return self.update_norm(grid_states + update)

    def propagate(self, patch_states):
        bsz, num_patches, dim = patch_states.shape
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise RuntimeError(f"Expected square patch grid, got {num_patches} patches")
        grid_states = patch_states.view(bsz, grid_size, grid_size, dim)
        global_context = patch_states.mean(dim=1)
        for _ in range(self.propagation_steps):
            grid_states = self.propagate_once(grid_states, global_context)
        return grid_states.reshape(bsz, num_patches, dim)

    def build_hypotheses(self, states, gate):
        context = states.mean(dim=1, keepdim=True).expand_as(states)
        logits = gate(torch.cat([states, context], dim=-1)).transpose(1, 2)
        weights = torch.softmax(logits, dim=-1)
        return weights @ states

    def compete(self, cls_state, hypotheses):
        global_hypothesis = hypotheses.mean(dim=1, keepdim=True).expand_as(hypotheses)
        other_context = (hypotheses.sum(dim=1, keepdim=True) - hypotheses) / max(1, self.num_hypotheses * 2 - 1)
        cls_context = cls_state.unsqueeze(1).expand_as(hypotheses)
        confidence_input = torch.cat(
            [hypotheses, global_hypothesis, other_context, hypotheses - other_context],
            dim=-1,
        )
        confidence = self.confidence(confidence_input).squeeze(-1)
        keep_weights = torch.softmax(confidence, dim=-1)
        suppress_weights = torch.softmax(-confidence, dim=-1)
        keep_summary = (keep_weights.unsqueeze(-1) * hypotheses).sum(dim=1)
        suppress_summary = (suppress_weights.unsqueeze(-1) * hypotheses).sum(dim=1)
        scale = self.suppress_scale(torch.cat([cls_state, keep_summary - suppress_summary], dim=-1))
        return keep_summary, suppress_summary, scale

    def forward(self, hidden_states, cls_index):
        cls_state = hidden_states[:, cls_index, :]
        patch_states = self.get_patch_states(hidden_states, cls_index)
        propagated_states = self.propagate(patch_states)
        hypotheses = torch.cat(
            [
                self.build_hypotheses(patch_states, self.hypothesis_gate),
                self.build_hypotheses(propagated_states, self.propagated_hypothesis_gate),
            ],
            dim=1,
        )
        keep_summary, suppress_summary, scale = self.compete(cls_state, hypotheses)
        update = self.keep_proj(keep_summary) - scale * self.suppress_proj(suppress_summary)
        return self.out_norm(cls_state + update)


class SoftDecoySuppressionTokenPooler(nn.Module):
    def __init__(self, dim, bottleneck_dim=128, dropout=0.1, num_hypotheses=4, propagation_steps=3):
        super().__init__()
        self.num_hypotheses = num_hypotheses
        self.propagation_steps = propagation_steps
        self.hypothesis_gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, num_hypotheses),
        )
        self.propagated_hypothesis_gate = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, num_hypotheses),
        )
        self.message_norm = nn.LayerNorm(dim * 3)
        self.message = nn.Sequential(
            nn.Linear(dim * 3, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )
        self.message_gate = nn.Sequential(
            nn.Linear(dim * 3, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
            nn.Sigmoid(),
        )
        self.update_norm = nn.LayerNorm(dim)
        self.confidence = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.keep_mask = nn.Sequential(
            nn.Linear(dim * 4, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.decoy_mask = nn.Sequential(
            nn.Linear(dim * 4, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )
        self.mix = nn.Sequential(
            nn.LayerNorm(dim * 4),
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def get_patch_states(self, hidden_states, cls_index):
        patch_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        patch_mask[:, cls_index] = False
        return hidden_states[patch_mask].view(
            hidden_states.shape[0],
            hidden_states.shape[1] - 1,
            hidden_states.shape[2],
        )

    def neighbor_states(self, grid_states):
        bsz, grid_size, _, dim = grid_states.shape
        zeros = torch.zeros(bsz, grid_size, 1, dim, device=grid_states.device, dtype=grid_states.dtype)
        left = torch.cat([zeros, grid_states[:, :, :-1, :]], dim=2)
        right = torch.cat([grid_states[:, :, 1:, :], zeros], dim=2)

        zeros = torch.zeros(bsz, 1, grid_size, dim, device=grid_states.device, dtype=grid_states.dtype)
        up = torch.cat([zeros, grid_states[:, :-1, :, :]], dim=1)
        down = torch.cat([grid_states[:, 1:, :, :], zeros], dim=1)
        return [left, right, up, down]

    def propagate_once(self, grid_states, global_context):
        global_context = global_context[:, None, None, :].expand_as(grid_states)
        updates = []
        for neighbor in self.neighbor_states(grid_states):
            message_input = torch.cat([grid_states, neighbor, global_context], dim=-1)
            message_input = self.message_norm(message_input)
            updates.append(self.message_gate(message_input) * self.message(message_input))
        update = torch.stack(updates, dim=0).mean(dim=0)
        return self.update_norm(grid_states + update)

    def propagate(self, patch_states):
        bsz, num_patches, dim = patch_states.shape
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size != num_patches:
            raise RuntimeError(f"Expected square patch grid, got {num_patches} patches")
        grid_states = patch_states.view(bsz, grid_size, grid_size, dim)
        global_context = patch_states.mean(dim=1)
        for _ in range(self.propagation_steps):
            grid_states = self.propagate_once(grid_states, global_context)
        return grid_states.reshape(bsz, num_patches, dim)

    def build_hypotheses(self, states, gate):
        context = states.mean(dim=1, keepdim=True).expand_as(states)
        logits = gate(torch.cat([states, context], dim=-1)).transpose(1, 2)
        weights = torch.softmax(logits, dim=-1)
        return weights @ states

    def compete(self, hypotheses):
        global_hypothesis = hypotheses.mean(dim=1, keepdim=True).expand_as(hypotheses)
        other_context = (hypotheses.sum(dim=1, keepdim=True) - hypotheses) / max(1, hypotheses.shape[1] - 1)
        confidence_input = torch.cat(
            [hypotheses, global_hypothesis, other_context, hypotheses - other_context],
            dim=-1,
        )
        confidence = self.confidence(confidence_input).squeeze(-1)
        keep_weights = torch.softmax(confidence, dim=-1)
        decoy_weights = torch.softmax(-confidence, dim=-1)
        keep_summary = (keep_weights.unsqueeze(-1) * hypotheses).sum(dim=1)
        decoy_summary = (decoy_weights.unsqueeze(-1) * hypotheses).sum(dim=1)
        return keep_summary, decoy_summary

    def masked_pool(self, states, keep_summary, decoy_summary, mask_head):
        keep_context = keep_summary.unsqueeze(1).expand_as(states)
        decoy_context = decoy_summary.unsqueeze(1).expand_as(states)
        mask_input = torch.cat([states, keep_context, decoy_context, states - decoy_context], dim=-1)
        logits = mask_head(mask_input).squeeze(-1)
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
        return (weights * states).sum(dim=1)

    def forward(self, hidden_states, cls_index):
        cls_state = hidden_states[:, cls_index, :]
        patch_states = self.get_patch_states(hidden_states, cls_index)
        propagated_states = self.propagate(patch_states)
        hypotheses = torch.cat(
            [
                self.build_hypotheses(patch_states, self.hypothesis_gate),
                self.build_hypotheses(propagated_states, self.propagated_hypothesis_gate),
            ],
            dim=1,
        )
        keep_summary, decoy_summary = self.compete(hypotheses)
        keep_pooled = self.masked_pool(patch_states, keep_summary, decoy_summary, self.keep_mask)
        propagated_pooled = self.masked_pool(propagated_states, keep_summary, decoy_summary, self.keep_mask)
        decoy_residual = self.masked_pool(patch_states, decoy_summary, keep_summary, self.decoy_mask)
        mixed = self.mix(torch.cat([cls_state, keep_pooled, propagated_pooled, keep_pooled - decoy_residual], dim=-1))
        return self.out_norm(cls_state + mixed)


class VimVisionWrapper(nn.Module):
    def __init__(
        self,
        model_name="vim_base",
        output_dim=512,
        pretrained=False,
        ckpt_path=None,
        path_adapter_layers=0,
        path_adapter_bottleneck=128,
        path_adapter_dropout=0.1,
        path_adapter_pooling="forward",
    ):
        super().__init__()

        self.input_resolution = 224

        if model_name == "vim_base":
            self.backbone = vim_base_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_middle_cls_token_div2(
                pretrained=pretrained,
                num_classes=0,
            )
            backbone_dim = 768
        else:
            raise ValueError(f"Unsupported Vim model: {model_name}")

        self.path_adapter_layers = int(path_adapter_layers)
        if self.path_adapter_layers < 0:
            raise ValueError("path_adapter_layers must be non-negative")
        if self.path_adapter_layers > 0:
            if self.path_adapter_layers > len(self.backbone.layers):
                raise ValueError(
                    f"path_adapter_layers={self.path_adapter_layers} exceeds "
                    f"Vim layers={len(self.backbone.layers)}"
                )
            self.path_adapters = nn.ModuleDict(
                {
                    str(idx): PathAwareStateAdapter(
                        backbone_dim,
                        bottleneck_dim=path_adapter_bottleneck,
                        dropout=path_adapter_dropout,
                    )
                    for idx in range(len(self.backbone.layers) - self.path_adapter_layers, len(self.backbone.layers))
                }
            )
            if path_adapter_pooling == "forward":
                self.path_pooler = PathAwareTokenPooler(
                    backbone_dim,
                    bottleneck_dim=path_adapter_bottleneck,
                    dropout=path_adapter_dropout,
                )
            elif path_adapter_pooling == "bidirectional":
                self.path_pooler = BidirectionalPathAwareTokenPooler(
                    backbone_dim,
                    bottleneck_dim=path_adapter_bottleneck,
                    dropout=path_adapter_dropout,
                )
            elif path_adapter_pooling == "start_bidirectional":
                self.path_pooler = StartAwareBidirectionalPathAwareTokenPooler(
                    backbone_dim,
                    bottleneck_dim=path_adapter_bottleneck,
                    dropout=path_adapter_dropout,
                )
            elif path_adapter_pooling == "geometry_bidirectional":
                self.path_pooler = GeometryAwareBidirectionalPathAwareTokenPooler(
                    backbone_dim,
                    bottleneck_dim=path_adapter_bottleneck,
                    dropout=path_adapter_dropout,
                )
            elif path_adapter_pooling == "path_state":
                self.path_pooler = PathStatePropagationTokenPooler(
                    backbone_dim,
                    bottleneck_dim=path_adapter_bottleneck,
                    dropout=path_adapter_dropout,
                )
            elif path_adapter_pooling == "hybrid_path_state":
                self.path_pooler = HybridPathStateBidirectionalTokenPooler(
                    backbone_dim,
                    bottleneck_dim=path_adapter_bottleneck,
                    dropout=path_adapter_dropout,
                )
            elif path_adapter_pooling == "gated_hybrid_path_state":
                self.path_pooler = GatedHybridPathStateTokenPooler(
                    backbone_dim,
                    bottleneck_dim=path_adapter_bottleneck,
                    dropout=path_adapter_dropout,
                )
            elif path_adapter_pooling == "omni_path_state":
                self.path_pooler = OmniPathStateTokenPooler(
                    backbone_dim,
                    bottleneck_dim=path_adapter_bottleneck,
                    dropout=path_adapter_dropout,
                )
            elif path_adapter_pooling == "memory_omni_path_state":
                self.path_pooler = MemoryOmniPathStateTokenPooler(
                    backbone_dim,
                    bottleneck_dim=path_adapter_bottleneck,
                    dropout=path_adapter_dropout,
                )
            elif path_adapter_pooling == "endpoint_omni_path_state":
                self.path_pooler = EndpointRefinementOmniTokenPooler(
                    backbone_dim,
                    bottleneck_dim=path_adapter_bottleneck,
                    dropout=path_adapter_dropout,
                )
            elif path_adapter_pooling == "layerwise_omni_path_state":
                self.path_pooler = LayerwiseOmniPathStateTokenPooler(
                    backbone_dim,
                    bottleneck_dim=path_adapter_bottleneck,
                    dropout=path_adapter_dropout,
                )
            elif path_adapter_pooling == "decoy_suppression":
                self.path_pooler = DecoySuppressionTokenPooler(
                    backbone_dim,
                    bottleneck_dim=path_adapter_bottleneck,
                    dropout=path_adapter_dropout,
                )
            elif path_adapter_pooling == "soft_decoy_suppression":
                self.path_pooler = SoftDecoySuppressionTokenPooler(
                    backbone_dim,
                    bottleneck_dim=path_adapter_bottleneck,
                    dropout=path_adapter_dropout,
                )
            else:
                raise ValueError(f"Unsupported path_adapter_pooling: {path_adapter_pooling}")
        else:
            self.path_adapters = nn.ModuleDict()
            self.path_pooler = None

        # 先建 wrapper 自己的層
        self.ln_post = nn.LayerNorm(backbone_dim)

        # # Residual + MLP
        # self.proj = ResidualProjectionHead(
        #     input_dim=backbone_dim,
        #     hidden_dim=1024,
        #     bottleneck_dim=768,
        #     output_dim=output_dim,
        #     dropout1=0.35,
        #     dropout2=0.55,
        # )

        # 改成 MLP
        self.proj = nn.Sequential(
            nn.Linear(backbone_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(1024, 768),
            nn.GELU(),
            nn.Dropout(0.55),
            nn.Linear(768, output_dim),
        )

        # # 舊版單層 Linear Projection
        # self.proj = nn.Linear(backbone_dim, output_dim, bias=False)

        # 再視情況載 backbone checkpoint
        if ckpt_path is not None:
            self.load_vim_checkpoint(ckpt_path)


    def load_vim_checkpoint(self, ckpt_path):
        print(f"Loading Vim checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state_dict = ckpt["model"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # 有些 checkpoint 會帶 module. 前綴
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            new_state_dict[k] = v
        state_dict = new_state_dict

        # 去掉分類 head，因為我們現在 num_classes=0，不需要 head
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("head."):
                continue
            filtered_state_dict[k] = v

        msg = self.backbone.load_state_dict(filtered_state_dict, strict=False)
        print("Vim backbone checkpoint loaded.")
        print("Unexpected keys:", msg.unexpected_keys)

    def forward(self, x):
        backbone_dtype = next(self.backbone.parameters()).dtype
        x = x.to(dtype=backbone_dtype)

        if self.path_adapter_layers > 0:
            x = self.forward_features_with_adapters(x)
        else:
            x = self.backbone.forward_features(x)   # [B, 768]
        x = self.ln_post(x)
        x = self.proj(x)                        # [B, output_dim]
        return x

    def forward_features_with_adapters(self, x):
        backbone = self.backbone
        if backbone.if_bidirectional or backbone.if_rope or backbone.flip_img_sequences_ratio > 0:
            raise RuntimeError("Path adapters currently support the configured non-RoPE non-bidirectional Vim backbone.")

        x = backbone.patch_embed(x)
        bsz, seq_len, _ = x.shape

        if backbone.if_cls_token:
            if backbone.use_double_cls_token:
                raise RuntimeError("Path adapters currently expect a single Vim class token.")
            cls_token = backbone.cls_token.expand(bsz, -1, -1)
            if backbone.use_middle_cls_token:
                token_position = seq_len // 2
                x = torch.cat((x[:, :token_position, :], cls_token, x[:, token_position:, :]), dim=1)
            else:
                token_position = 0
                x = torch.cat((cls_token, x), dim=1)

        if backbone.if_abs_pos_embed:
            x = x + backbone.pos_embed
            x = backbone.pos_drop(x)

        residual = None
        hidden_states = x
        layerwise_states = []
        for idx, layer in enumerate(backbone.layers):
            hidden_states, residual = layer(hidden_states, residual, inference_params=None)
            adapter_key = str(idx)
            if adapter_key in self.path_adapters:
                hidden_states = self.path_adapters[adapter_key](hidden_states)
                if getattr(self.path_pooler, "wants_layerwise_states", False):
                    layerwise_states.append(hidden_states)

        if not backbone.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + backbone.drop_path(hidden_states)
            hidden_states = backbone.norm_f(residual.to(dtype=backbone.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(backbone.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                backbone.drop_path(hidden_states),
                backbone.norm_f.weight,
                backbone.norm_f.bias,
                eps=backbone.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=backbone.residual_in_fp32,
            )

        if backbone.if_cls_token:
            if self.path_pooler is not None:
                if getattr(self.path_pooler, "wants_layerwise_states", False):
                    return self.path_pooler(hidden_states, token_position, layerwise_states)
                return self.path_pooler(hidden_states, token_position)
            return hidden_states[:, token_position, :]
        if backbone.final_pool_type == "none":
            return hidden_states[:, -1, :]
        if backbone.final_pool_type == "mean":
            return hidden_states.mean(dim=1)
        if backbone.final_pool_type == "max":
            return hidden_states
        if backbone.final_pool_type == "all":
            return hidden_states
        raise NotImplementedError
