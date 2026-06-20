"""The STILL per-layer Perceiver (basic, standard init).

Learnable latents cross-attend into each layer's full ``[K;V]``, self-attend among
themselves, and project to compact K, V, and a per-position attention-bias scalar.
One ``STILLPerceiver`` holds one ``STILLLayerPerceiver`` per transformer layer.

Shapes: per layer the input K/V are ``[H_kv, T, D]`` (one document, batch squeezed);
outputs are ``compact_K, compact_V == [H_kv, t, D]`` and ``bias == [H_kv, t]``.

v1 uses standard ``nn.Linear`` init throughout. The paper's identity init +
content-stripping biases (one of the three correctness fixes) is deferred.
"""

from __future__ import annotations

import torch
from torch import nn


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Single-head scaled dot-product attention. q:[...,n,d] k,v:[...,m,d] -> [...,n,d]."""
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    attn = scores.softmax(dim=-1)
    return torch.matmul(attn, v)


class PerceiverBlock(nn.Module):
    """cross-attn (latents query into kv) -> self-attn among latents -> MLP, pre-norm + residual."""

    def __init__(self, latent_dim: int, mlp_ratio: int = 4):
        super().__init__()
        # cross-attention (1 head)
        self.cross_norm_q = nn.LayerNorm(latent_dim)
        self.cross_norm_kv = nn.LayerNorm(latent_dim)
        self.cross_q = nn.Linear(latent_dim, latent_dim)
        self.cross_k = nn.Linear(latent_dim, latent_dim)
        self.cross_v = nn.Linear(latent_dim, latent_dim)
        self.cross_out = nn.Linear(latent_dim, latent_dim)
        # self-attention (1 head)
        self.self_norm = nn.LayerNorm(latent_dim)
        self.self_q = nn.Linear(latent_dim, latent_dim)
        self.self_k = nn.Linear(latent_dim, latent_dim)
        self.self_v = nn.Linear(latent_dim, latent_dim)
        self.self_out = nn.Linear(latent_dim, latent_dim)
        # MLP
        self.mlp_norm = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, mlp_ratio * latent_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * latent_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # z: [H_kv, t, d_lat], kv: [H_kv, T, d_lat]
        zq = self.cross_norm_q(z)
        kvn = self.cross_norm_kv(kv)
        attn = _attend(self.cross_q(zq), self.cross_k(kvn), self.cross_v(kvn))
        z = z + self.cross_out(attn)

        zn = self.self_norm(z)
        attn = _attend(self.self_q(zn), self.self_k(zn), self.self_v(zn))
        z = z + self.self_out(attn)

        z = z + self.mlp(self.mlp_norm(z))
        return z


class STILLLayerPerceiver(nn.Module):
    """The Perceiver for a single transformer layer."""

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        num_latents: int,
        latent_dim: int,
        num_blocks: int,
    ):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_latents = num_latents
        # learnable latents per KV-head group: [H_kv, t, d_lat]
        self.latents = nn.Parameter(torch.randn(num_kv_heads, num_latents, latent_dim) * 0.02)
        self.in_proj = nn.Linear(2 * head_dim, latent_dim)
        self.blocks = nn.ModuleList(PerceiverBlock(latent_dim) for _ in range(num_blocks))
        self.k_head = nn.Linear(latent_dim, head_dim)
        self.v_head = nn.Linear(latent_dim, head_dim)
        self.bias_head = nn.Linear(latent_dim, 1)

    def forward(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # key, value: [H_kv, T, D]
        kv = self.in_proj(torch.cat([key, value], dim=-1))  # [H_kv, T, d_lat]
        z = self.latents  # [H_kv, t, d_lat]
        for blk in self.blocks:
            z = blk(z, kv)
        compact_k = self.k_head(z)  # [H_kv, t, D]
        compact_v = self.v_head(z)  # [H_kv, t, D]
        bias = self.bias_head(z).squeeze(-1)  # [H_kv, t]
        return compact_k, compact_v, bias


class STILLPerceiver(nn.Module):
    """Holds one ``STILLLayerPerceiver`` per transformer layer."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_latents: int,
        latent_dim: int,
        num_blocks: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_latents = num_latents
        self.layers = nn.ModuleList(
            STILLLayerPerceiver(num_kv_heads, head_dim, num_latents, latent_dim, num_blocks)
            for _ in range(num_layers)
        )

    def forward_layer(
        self, layer_idx: int, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.layers[layer_idx](key, value)

    @classmethod
    def from_model_config(cls, model_config, cfg) -> "STILLPerceiver":
        """Build a perceiver sized to a HuggingFace model config + a ``STILLConfig``."""
        head_dim = getattr(
            model_config,
            "head_dim",
            model_config.hidden_size // model_config.num_attention_heads,
        )
        return cls(
            num_layers=model_config.num_hidden_layers,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=head_dim,
            num_latents=cfg.num_latents,
            latent_dim=cfg.latent_dim,
            num_blocks=cfg.num_blocks,
        )
