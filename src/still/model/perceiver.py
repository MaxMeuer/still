"""STILL per-layer Perceiver with the three correctness fixes from the paper.

Fix 1: RoPE un-rotate/re-rotate pipeline — strips positional encoding before
       compression, uses perceiver-internal RoPE for position-aware routing,
       re-applies model RoPE to compact keys at evenly-spaced positions.

Fix 2: No final RMSNorm before output heads — real KV entries have varying
       norms that carry information.

Fix 3: Identity initialization with attention biases — the perceiver starts as
       a near-identity pass-through so every latent copies its positionally-
       nearest input. Scales monotonically from 128 to 8192 latents.
"""

from __future__ import annotations

import math

import torch
from torch import nn


# ─── RoPE utilities ───────────────────────────────────────────────────────────


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cache(
    positions: torch.Tensor,
    dim: int,
    rope_theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin for given positions. Returns shapes broadcastable to [..., seq, dim]."""
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    # positions: [T] → freqs: [T, dim//2]
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)  # [T, dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Forward RoPE: x has shape [..., T, dim], cos/sin have shape [T, dim]."""
    return (x * cos) + (_rotate_half(x) * sin)


def apply_inverse_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Inverse RoPE (strip positional encoding)."""
    return (x * cos) - (_rotate_half(x) * sin)


# ─── Perceiver block ─────────────────────────────────────────────────────────


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Single-head scaled dot-product attention. q:[...,n,d] k,v:[...,m,d] -> [...,n,d]."""
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    attn = scores.softmax(dim=-1)
    return torch.matmul(attn, v)


class PerceiverBlock(nn.Module):
    """Cross-attn → self-attn → MLP with pre-norm + residual.

    Cross-attention Q and K projections have bias (for identity init).
    Cross-attention accepts optional RoPE cos/sin for position-aware routing.
    No normalization on the KV input (value pathway must be identity at init).
    """

    def __init__(self, latent_dim: int, mlp_ratio: int = 4):
        super().__init__()
        # cross-attention (1 head) — bias on Q/K for identity init
        self.cross_norm_q = nn.LayerNorm(latent_dim)
        self.cross_q = nn.Linear(latent_dim, latent_dim, bias=True)
        self.cross_k = nn.Linear(latent_dim, latent_dim, bias=True)
        self.cross_v = nn.Linear(latent_dim, latent_dim, bias=False)
        self.cross_out = nn.Linear(latent_dim, latent_dim, bias=False)
        # self-attention (1 head)
        self.self_norm = nn.LayerNorm(latent_dim)
        self.self_q = nn.Linear(latent_dim, latent_dim)
        self.self_k = nn.Linear(latent_dim, latent_dim)
        self.self_v = nn.Linear(latent_dim, latent_dim)
        self.self_out = nn.Linear(latent_dim, latent_dim, bias=False)
        # MLP
        self.mlp_norm = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, mlp_ratio * latent_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * latent_dim, latent_dim),
        )

    def forward(
        self,
        z: torch.Tensor,
        kv: torch.Tensor,
        cross_cos_q: torch.Tensor | None = None,
        cross_sin_q: torch.Tensor | None = None,
        cross_cos_k: torch.Tensor | None = None,
        cross_sin_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # z: [H_kv, t, d_lat], kv: [H_kv, T, d_lat]
        zq = self.cross_norm_q(z)
        q = self.cross_q(zq)
        k = self.cross_k(kv)
        v = self.cross_v(kv)

        # perceiver-internal RoPE on cross-attention Q/K
        if cross_cos_q is not None:
            q = apply_rope(q, cross_cos_q, cross_sin_q)
            k = apply_rope(k, cross_cos_k, cross_sin_k)

        attn = _attend(q, k, v)
        z = z + self.cross_out(attn)

        zn = self.self_norm(z)
        attn = _attend(self.self_q(zn), self.self_k(zn), self.self_v(zn))
        z = z + self.self_out(attn)

        z = z + self.mlp(self.mlp_norm(z))
        return z


# ─── Layer perceiver ──────────────────────────────────────────────────────────


class STILLLayerPerceiver(nn.Module):
    """Perceiver for a single transformer layer with all three correctness fixes."""

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        num_latents: int,
        latent_dim: int,
        num_blocks: int,
        rope_theta: float = 1_000_000.0,
    ):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.rope_theta = rope_theta

        # learnable latents per KV-head group: [H_kv, t, d_lat]
        self.latents = nn.Parameter(torch.randn(num_kv_heads, num_latents, latent_dim) * 0.02)
        self.in_proj = nn.Linear(2 * head_dim, latent_dim, bias=False)
        self.blocks = nn.ModuleList(PerceiverBlock(latent_dim) for _ in range(num_blocks))
        self.k_head = nn.Linear(latent_dim, head_dim, bias=False)
        self.v_head = nn.Linear(latent_dim, head_dim, bias=False)
        self.bias_head = nn.Linear(latent_dim, 1, bias=True)

        self._identity_init()

    def _identity_init(self):
        """Initialize as near-identity: each latent copies its nearest input position."""
        d_lat = self.latent_dim
        d = self.head_dim

        assert d_lat == 2 * d, f"Identity init requires latent_dim == 2*head_dim, got {d_lat} vs {2*d}"

        # --- Value pathway: identity chain ---
        # in_proj: [K;V] (2d) → latent (2d) = identity
        nn.init.eye_(self.in_proj.weight)

        # k_head extracts first half (key portion): weight shape (head_dim, latent_dim)
        with torch.no_grad():
            self.k_head.weight.zero_()
            self.k_head.weight[:, :d] = torch.eye(d)

        # v_head extracts second half (value portion)
        with torch.no_grad():
            self.v_head.weight.zero_()
            self.v_head.weight[:, d:] = torch.eye(d)

        # bias_head: zero init (bias starts at 0)
        with torch.no_grad():
            self.bias_head.weight.zero_()
            self.bias_head.bias.zero_()

        # --- Per-block initialization ---
        q_hat = torch.ones(d_lat) / math.sqrt(d_lat)

        for block_idx, block in enumerate(self.blocks):
            with torch.no_grad():
                if block_idx == 0:
                    # First block: cross_v and cross_out are identity
                    nn.init.eye_(block.cross_v.weight)
                    nn.init.eye_(block.cross_out.weight)
                else:
                    # Later blocks: zero cross_out so they're inactive at init
                    block.cross_out.weight.zero_()
                    block.cross_v.weight.zero_()

                # Q/K biases for content-stripping
                block.cross_q.bias.copy_(q_hat)
                block.cross_k.bias.copy_(q_hat * 10.0)
                # Q/K weights: zero so only bias matters at init
                block.cross_q.weight.zero_()
                block.cross_k.weight.zero_()

                # Self-attention output: zero in all blocks
                block.self_out.weight.zero_()

                # MLP final layer: zero
                block.mlp[-1].weight.zero_()
                if block.mlp[-1].bias is not None:
                    block.mlp[-1].bias.zero_()

    def forward(
        self, key: torch.Tensor, value: torch.Tensor, seq_len: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            key: [H_kv, T, D] — post-RoPE keys from the frozen model's cache
            value: [H_kv, T, D] — values from the frozen model's cache
            seq_len: T (passed explicitly; defaults to key.shape[-2])
        """
        T = seq_len if seq_len is not None else key.shape[-2]
        device = key.device
        dtype = key.dtype

        # ─── Fix 1, Step 1: Un-rotate (strip model's RoPE from cached keys) ───
        positions_full = torch.arange(T, device=device, dtype=dtype)
        cos_full, sin_full = build_rope_cache(positions_full, self.head_dim, self.rope_theta, device, dtype)
        key = apply_inverse_rope(key, cos_full, sin_full)  # [H_kv, T, D] now position-free

        # ─── Concatenate K;V and project to latent space ───
        kv = self.in_proj(torch.cat([key, value], dim=-1))  # [H_kv, T, d_lat]

        # ─── Fix 1, Step 2: Perceiver-internal RoPE for cross-attention ───
        # Key positions = original token positions [0, T)
        cross_cos_k, cross_sin_k = build_rope_cache(
            positions_full, self.latent_dim, self.rope_theta, device, dtype
        )
        # Query (latent) positions = evenly spaced across [0, T-1]
        positions_latent = torch.linspace(0, T - 1, self.num_latents, device=device, dtype=dtype)
        cross_cos_q, cross_sin_q = build_rope_cache(
            positions_latent, self.latent_dim, self.rope_theta, device, dtype
        )

        # ─── Run perceiver blocks ───
        z = self.latents  # [H_kv, t, d_lat]
        for i, blk in enumerate(self.blocks):
            if i == 0:
                z = blk(z, kv, cross_cos_q, cross_sin_q, cross_cos_k, cross_sin_k)
            else:
                z = blk(z, kv)

        # ─── Output heads (no final norm — Fix 2) ───
        compact_k = self.k_head(z)  # [H_kv, t, D]
        compact_v = self.v_head(z)  # [H_kv, t, D]
        bias = self.bias_head(z).squeeze(-1)  # [H_kv, t]

        # ─── Fix 1, Step 3: Re-rotate compact keys at evenly-spaced positions ───
        cos_latent, sin_latent = build_rope_cache(
            positions_latent, self.head_dim, self.rope_theta, device, dtype
        )
        compact_k = apply_rope(compact_k, cos_latent, sin_latent)

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
        rope_theta: float = 1_000_000.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_latents = num_latents
        self.layers = nn.ModuleList(
            STILLLayerPerceiver(
                num_kv_heads, head_dim, num_latents, latent_dim, num_blocks, rope_theta
            )
            for _ in range(num_layers)
        )

    def forward_layer(
        self, layer_idx: int, key: torch.Tensor, value: torch.Tensor, seq_len: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.layers[layer_idx](key, value, seq_len)

    @classmethod
    def from_model_config(cls, model_config, cfg) -> "STILLPerceiver":
        """Build a perceiver sized to a HuggingFace model config + a ``STILLConfig``."""
        head_dim = getattr(
            model_config,
            "head_dim",
            model_config.hidden_size // model_config.num_attention_heads,
        )
        rope_theta = getattr(model_config, "rope_theta", 1_000_000.0)
        return cls(
            num_layers=model_config.num_hidden_layers,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=head_dim,
            num_latents=cfg.num_latents,
            latent_dim=cfg.latent_dim,
            num_blocks=cfg.num_blocks,
            rope_theta=rope_theta,
        )
