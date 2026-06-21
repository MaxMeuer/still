from __future__ import annotations

import torch

from still.config import STILLConfig
from still.model.attention import CompactCache
from still.model.wrapper import STILLModel


def _tiny(tiny_model_path):
    cfg = STILLConfig(model_name=tiny_model_path, num_latents=4, latent_dim=16, num_blocks=2)
    return STILLModel(tiny_model_path, cfg=cfg, device="cpu")


def test_compact_tokens_and_extend(tiny_model_path):
    model = _tiny(tiny_model_path)
    a = model.compact_tokens(torch.randint(0, 100, (1, 20)))
    b = model.compact_tokens(torch.randint(0, 100, (1, 20)))
    assert isinstance(a, CompactCache) and a.num_layers == model.base.config.num_hidden_layers
    t0 = a.num_latents
    a.extend(b)
    # two chunks accumulate compact positions
    assert a.num_latents == 2 * t0
    assert a.bias[0].shape[-1] == 2 * t0


def test_generate_compacted_bounded_and_finite(tiny_model_path):
    model = _tiny(tiny_model_path)
    # long prompt that must be compacted: threshold small, chunk small
    prompt = torch.randint(0, 100, (1, 120))
    out = model.generate_compacted(
        prompt,
        max_new_tokens=5,
        threshold=48, live_window=16, compaction_chunk=16, do_sample=False,
    )
    assert isinstance(out, list)
    assert 0 < len(out) <= 5
    assert all(isinstance(t, int) for t in out)


def test_generate_compacted_no_compaction_path(tiny_model_path):
    model = _tiny(tiny_model_path)
    # short prompt under threshold -> plain generate, no compaction
    prompt = torch.randint(0, 100, (1, 10))
    out = model.generate_compacted(
        prompt, max_new_tokens=4, threshold=4096, compaction_chunk=2048, do_sample=False
    )
    assert 0 < len(out) <= 4


def test_server_render_and_generate_chat(tiny_model_path, tokenizer):
    """The server path: chat template -> clean list[int] -> compacted generation."""
    from still.serve.server import _render_prompt

    model = _tiny(tiny_model_path)
    messages = [{"role": "user", "content": "hello " * 300 + "what is 2+2?"}]
    ids = _render_prompt(tokenizer, messages, enable_thinking=False)
    assert isinstance(ids, list) and all(isinstance(x, int) for x in ids)
    out = model.generate_compacted(
        ids, max_new_tokens=4, threshold=48, live_window=16, compaction_chunk=16, do_sample=False
    )
    assert 0 < len(out) <= 4
