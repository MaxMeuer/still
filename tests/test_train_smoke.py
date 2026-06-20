from __future__ import annotations

import os

import torch
from datasets import Dataset

from still.config import STILLConfig
from still.train import run_training


def _make_dataset(path, n=2):
    torch.manual_seed(0)
    rows = {
        "doc_input_ids": [torch.randint(0, 100, (12,)).tolist() for _ in range(n)],
        "query_input_ids": [torch.randint(0, 100, (5,)).tolist() for _ in range(n)],
        "answer_input_ids": [torch.randint(0, 100, (3,)).tolist() for _ in range(n)],
        "gold_idx": [0] * n,
        "hard": [False] * n,
    }
    Dataset.from_dict(rows).save_to_disk(path)
    return path


def test_train_smoke_decreases_and_checkpoints(tmp_path, tiny_model_path):
    ds_path = _make_dataset(str(tmp_path / "ds"))
    ckpt_dir = str(tmp_path / "ckpt")
    cfg = STILLConfig(
        model_name=tiny_model_path, num_latents=4, latent_dim=16, num_blocks=2, device="cpu"
    )

    # snapshot a base tensor to confirm it is unchanged after training
    from still.model.wrapper import STILLModel

    probe = STILLModel(tiny_model_path, cfg=cfg, device="cpu")
    base_before = probe.base.model.layers[0].self_attn.q_proj.weight.detach().clone()
    del probe

    out = run_training(
        dataset_path=ds_path,
        model_name=tiny_model_path,
        cfg=cfg,
        steps=6,
        lr=1e-2,
        ckpt_dir=ckpt_dir,
        ckpt_every=0,
        seed=0,
    )
    losses = out["losses"]
    assert len(losses) == 6
    assert all(torch.isfinite(torch.tensor(losses)))
    # loss decreases over training (final clearly below first)
    assert losses[-1] < losses[0]
    # checkpoint written
    assert os.path.exists(out["ckpt"])

    # base params unchanged
    after = STILLModel(tiny_model_path, cfg=cfg, device="cpu")
    base_after = after.base.model.layers[0].self_attn.q_proj.weight.detach()
    assert torch.allclose(base_before, base_after)
