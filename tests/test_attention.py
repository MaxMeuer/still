from __future__ import annotations

from types import SimpleNamespace

import torch

from still.config import STILLConfig
from still.model.attention import still_attention
from still.model.wrapper import STILLModel


def test_still_attention_matches_eager_when_no_bias():
    from transformers.models.qwen3.modeling_qwen3 import (
        eager_attention_forward,
        repeat_kv,  # noqa: F401  (imported to ensure the symbol exists)
    )

    torch.manual_seed(0)
    B, H_q, H_kv, q, k, D = 1, 4, 2, 5, 5, 8
    n_rep = H_q // H_kv
    query = torch.randn(B, H_q, q, D)
    key = torch.randn(B, H_kv, k, D)
    value = torch.randn(B, H_kv, k, D)
    # additive causal mask [B,1,q,k]
    mask = torch.triu(torch.full((q, k), float("-inf")), diagonal=1)[None, None]
    scaling = D**-0.5

    module = SimpleNamespace(num_key_value_groups=n_rep, training=False)
    # no _still_bias attribute -> additive-only path is a no-op
    out_still, _ = still_attention(module, query, key, value, mask, scaling=scaling)
    out_eager, _ = eager_attention_forward(module, query, key, value, mask, scaling=scaling)
    assert torch.allclose(out_still, out_eager, atol=1e-5), (out_still - out_eager).abs().max()


def test_still_attention_zero_bias_is_noop():
    torch.manual_seed(1)
    B, H_q, H_kv, q, k, D = 1, 4, 2, 3, 3, 8
    n_rep = H_q // H_kv
    query = torch.randn(B, H_q, q, D)
    key = torch.randn(B, H_kv, k, D)
    value = torch.randn(B, H_kv, k, D)
    scaling = D**-0.5

    m_none = SimpleNamespace(num_key_value_groups=n_rep, training=False)
    m_zero = SimpleNamespace(num_key_value_groups=n_rep, training=False, _still_bias=torch.zeros(H_kv, k))
    out_none, _ = still_attention(m_none, query, key, value, None, scaling=scaling)
    out_zero, _ = still_attention(m_zero, query, key, value, None, scaling=scaling)
    assert torch.allclose(out_none, out_zero, atol=1e-6)


def _tiny_still_model(tiny_model_path):
    cfg = STILLConfig(model_name=tiny_model_path, num_latents=4, latent_dim=16, num_blocks=2)
    return STILLModel(tiny_model_path, cfg=cfg, device="cpu")


def test_decode_returns_finite_answer_logits(tiny_model_path):
    model = _tiny_still_model(tiny_model_path)
    doc = torch.randint(0, 100, (1, 12))
    query = torch.randint(0, 100, (1, 5))
    answer = torch.randint(0, 100, (1, 3))
    cache = model.compress(doc)
    logits = model.decode(query, answer, cache)
    assert logits.shape == (3, model.base.config.vocab_size)
    assert torch.isfinite(logits).all()


def test_gradient_isolation_to_perceiver(tiny_model_path):
    model = _tiny_still_model(tiny_model_path)
    doc = torch.randint(0, 100, (1, 12))
    query = torch.randint(0, 100, (1, 5))
    answer = torch.randint(0, 100, (1, 3))

    cache = model.compress(doc)
    logits = model.decode(query, answer, cache)
    logits.sum().backward()

    # every base param has no grad
    for name, p in model.base.named_parameters():
        assert p.grad is None, f"base param {name} received a gradient"
    # every perceiver param has a grad
    for name, p in model.perceiver.named_parameters():
        assert p.grad is not None, f"perceiver param {name} got no gradient"


def test_still_attention_is_invoked_during_decode(tiny_model_path):
    from still.model import attention as attn_mod

    model = _tiny_still_model(tiny_model_path)
    doc = torch.randint(0, 100, (1, 12))
    query = torch.randint(0, 100, (1, 5))
    answer = torch.randint(0, 100, (1, 3))
    cache = model.compress(doc)

    attn_mod.reset_invocation_count()
    model.decode(query, answer, cache)
    # one call per layer
    assert attn_mod.invocation_count() == model.base.config.num_hidden_layers
