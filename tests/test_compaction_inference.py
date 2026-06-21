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


def test_recompact_keeps_constant_budget(tiny_model_path):
    """Recursive compaction holds the cache at num_latents across N passes (constant memory)."""
    model = _tiny(tiny_model_path)
    budget = model.cfg.num_latents  # 4 for the tiny config
    cache = model.compact_tokens(torch.randint(0, 100, (20,)).tolist())
    assert cache.num_latents == budget
    for _ in range(8):  # 1,2,...,8 passes
        cache = model.recompact(cache, torch.randint(0, 100, (16,)).tolist())
        assert cache.num_latents == budget  # never grows
        assert cache.num_layers == model.base.config.num_hidden_layers


def test_generate_against_recursive_cache(tiny_model_path):
    model = _tiny(tiny_model_path)
    cache = model.compact_tokens(torch.randint(0, 100, (20,)).tolist())
    cache = model.recompact(cache, torch.randint(0, 100, (16,)).tolist())
    out = model.decode_generate(
        torch.randint(0, 100, (8,)).tolist(), cache, max_new_tokens=3, do_sample=False
    )
    assert 0 < len(out) <= 3
    assert all(isinstance(t, int) for t in out)


def test_incremental_compaction_reuses_blocks(tiny_model_path):
    """Growing conversation: only new tokens get compacted each turn, blocks accumulate."""
    import still.serve.server as srv

    model = _tiny(tiny_model_path)
    srv._CONV = {"tokens": None, "blocks": None, "n": 0}
    base = torch.randint(0, 100, (200,)).tolist()

    cache1, live1 = srv._incremental_compact(model, base, threshold=48, live_window=16, chunk=16)
    n1 = srv._CONV["n"]
    assert cache1 is not None and n1 > 0 and len(live1) <= 16 + 16

    # extend the conversation (same prefix + new tokens) -> n grows, prior blocks reused
    grown = base + torch.randint(0, 100, (64,)).tolist()
    cache2, live2 = srv._incremental_compact(model, grown, threshold=48, live_window=16, chunk=16)
    assert srv._CONV["n"] >= n1  # compacted further, did not reset to 0
    # generation works against the accumulated cache
    out = model.decode_generate(live2, cache2, max_new_tokens=3, do_sample=False)
    assert 0 < len(out) <= 3


def test_server_recursive_vs_append_growth(tiny_model_path):
    """recursive mode holds the cache at num_latents across a growing conversation; append grows."""
    import still.serve.server as srv

    model = _tiny(tiny_model_path)
    budget = model.cfg.num_latents
    base = torch.randint(0, 100, (64,)).tolist()
    grown = base + torch.randint(0, 100, (96,)).tolist()

    # recursive: constant budget across turns
    srv._CONV = {"tokens": None, "blocks": None, "n": 0}
    srv._incremental_compact(model, base, 16, 16, 16, mode="recursive")
    c_rec, _ = srv._incremental_compact(model, grown, 16, 16, 16, mode="recursive")
    assert c_rec.num_latents == budget

    # append: cache grows past the budget
    srv._CONV = {"tokens": None, "blocks": None, "n": 0}
    srv._incremental_compact(model, base, 16, 16, 16, mode="append")
    c_app, _ = srv._incremental_compact(model, grown, 16, 16, 16, mode="append")
    assert c_app.num_latents > budget
