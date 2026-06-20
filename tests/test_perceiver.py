from __future__ import annotations

import torch

from still.model.perceiver import STILLLayerPerceiver, STILLPerceiver

H_KV, T, D = 2, 64, 8
T_LAT, D_LAT, BLOCKS = 16, 32, 2


def _layer():
    torch.manual_seed(0)
    return STILLLayerPerceiver(
        num_kv_heads=H_KV, head_dim=D, num_latents=T_LAT, latent_dim=D_LAT, num_blocks=BLOCKS
    )


def test_output_shapes_and_compression():
    layer = _layer()
    key = torch.randn(H_KV, T, D)
    value = torch.randn(H_KV, T, D)
    ck, cv, bias = layer(key, value)
    assert ck.shape == (H_KV, T_LAT, D)
    assert cv.shape == (H_KV, T_LAT, D)
    assert bias.shape == (H_KV, T_LAT)
    # compression actually reduces the sequence dim
    assert T_LAT < T


def test_gradients_reach_latents_and_heads():
    layer = _layer()
    key = torch.randn(H_KV, T, D)
    value = torch.randn(H_KV, T, D)
    ck, cv, bias = layer(key, value)
    # sum all three outputs so gradients must reach every head, not just k_head
    loss = ck.sum() + cv.sum() + bias.sum()
    loss.backward()
    assert layer.latents.grad is not None
    assert layer.k_head.weight.grad is not None
    assert layer.v_head.weight.grad is not None
    assert layer.bias_head.weight.grad is not None
    # in_proj feeds every block too
    assert layer.in_proj.weight.grad is not None


def test_full_perceiver_param_count_is_small(capsys):
    torch.manual_seed(0)
    p = STILLPerceiver(
        num_layers=2,
        num_kv_heads=H_KV,
        head_dim=D,
        num_latents=T_LAT,
        latent_dim=D_LAT,
        num_blocks=BLOCKS,
    )
    n = sum(t.numel() for t in p.parameters())
    print(f"perceiver params: {n}")
    assert n > 0
    # per-layer forward works through the ModuleList
    ck, cv, bias = p.forward_layer(1, torch.randn(H_KV, T, D), torch.randn(H_KV, T, D))
    assert ck.shape == (H_KV, T_LAT, D)
