"""Registered biased attention forward + the compact KV cache.

The decode-against-compact-cache reduces to: compact K/V supplied as ``past_key_values``
plus an additive per-position bias on the attention scores over the compact (doc) key
positions. No paged-attention kernel.

The custom forward is registered through the transformers attention-implementation
registry (``AttentionInterface.register``) and selected via
``config._attn_implementation = "still"``. The per-layer compact bias is read off the
attention ``module`` itself (``module._still_bias``), because the registered callback
only receives ``module`` (not the cache), and the bias differs per layer.

Signature note (verified against transformers 5.12.1): the interface is called as
``fn(module, query, key, value, attention_mask, dropout=..., scaling=..., **kwargs)``
and ``key``/``value`` are the *raw* KV-head tensors (``[B, H_kv, k, D]``) that the
forward must ``repeat_kv`` itself.
"""

from __future__ import annotations

import torch
from torch import nn

STILL_ATTENTION_NAME = "still"

# Counter incremented every time the registered forward runs, so a manual decode can
# assert the custom attention is actually invoked.
_INVOCATION_COUNT = 0


def reset_invocation_count() -> None:
    global _INVOCATION_COUNT
    _INVOCATION_COUNT = 0


def invocation_count() -> int:
    return _INVOCATION_COUNT


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand KV heads to query heads (matches transformers' repeat_kv)."""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def still_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs,
):
    """Eager attention plus an additive bias over the compact (doc) key positions.

    With ``module._still_bias`` unset (or None) this is bit-for-bit stock eager
    attention, so the bias path is additive-only.
    """
    global _INVOCATION_COUNT
    _INVOCATION_COUNT += 1

    n_rep = getattr(module, "num_key_value_groups", 1)
    key_states = repeat_kv(key, n_rep)
    value_states = repeat_kv(value, n_rep)

    if scaling is None:
        scaling = query.shape[-1] ** -0.5

    scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling  # [B, H_q, q, k]

    bias = getattr(module, "_still_bias", None)  # [H_kv, t] over the compact prefix
    if bias is not None:
        t = bias.shape[-1]
        k_len = scores.shape[-1]
        # expand bias from KV heads to query heads, pad zeros over live positions
        bias_q = bias.repeat_interleave(n_rep, dim=0)  # [H_q, t]
        if t < k_len:
            pad = bias_q.new_zeros(bias_q.shape[0], k_len - t)
            bias_q = torch.cat([bias_q, pad], dim=-1)  # [H_q, k_len]
        bias_q = bias_q[None, :, None, :]  # [1, H_q, 1, k_len]
        scores = scores + bias_q

    if attention_mask is not None:
        # the causal/prefix mask covers the live region; slice to the key length
        scores = scores + attention_mask[..., : scores.shape[-1]]

    attn = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    attn = nn.functional.dropout(attn, p=dropout, training=module.training)
    attn_output = torch.matmul(attn, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def register_still_attention() -> None:
    """Register ``still_attention`` under the name ``"still"`` (idempotent)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface

    if STILL_ATTENTION_NAME not in ALL_ATTENTION_FUNCTIONS:
        AttentionInterface.register(STILL_ATTENTION_NAME, still_attention)


class CompactCache:
    """Per-layer compact K/V/bias produced by the perceiver.

    ``to_dynamic_cache`` materializes a ``DynamicCache`` prefilled with the compact K/V
    (grad-carrying) that a base-model forward appends live query/answer K/V onto. The
    per-layer bias is applied separately by attaching it to the attention modules.
    """

    def __init__(self) -> None:
        self.compact_k: list[torch.Tensor] = []  # each [H_kv, t, D]
        self.compact_v: list[torch.Tensor] = []
        self.bias: list[torch.Tensor] = []  # each [H_kv, t]

    def add(self, compact_k: torch.Tensor, compact_v: torch.Tensor, bias: torch.Tensor) -> None:
        self.compact_k.append(compact_k)
        self.compact_v.append(compact_v)
        self.bias.append(bias)

    def extend(self, other: "CompactCache") -> None:
        """Append another compacted chunk: concat per layer along the latent dim.

        Used for chunked inference-time compaction — each compressed chunk adds ``t``
        more compact positions per layer (K/V along dim -2, bias along dim -1).
        """
        if not self.compact_k:
            self.compact_k = [t for t in other.compact_k]
            self.compact_v = [t for t in other.compact_v]
            self.bias = [t for t in other.bias]
            return
        for i in range(len(self.compact_k)):
            self.compact_k[i] = torch.cat([self.compact_k[i], other.compact_k[i]], dim=-2)
            self.compact_v[i] = torch.cat([self.compact_v[i], other.compact_v[i]], dim=-2)
            self.bias[i] = torch.cat([self.bias[i], other.bias[i]], dim=-1)

    @property
    def num_layers(self) -> int:
        return len(self.compact_k)

    @property
    def num_latents(self) -> int:
        return self.compact_k[0].shape[-2]

    def to_dynamic_cache(self):
        from transformers import DynamicCache

        cache = DynamicCache()
        for i in range(self.num_layers):
            k = self.compact_k[i].unsqueeze(0)  # [1, H_kv, t, D]
            v = self.compact_v[i].unsqueeze(0)
            cache.update(k, v, i)
        return cache
