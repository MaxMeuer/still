from __future__ import annotations

import torch
from datasets import Dataset

from still.config import STILLConfig
from still.eval import run_eval
from still.train import run_training


def _make_dataset(path, n=4):
    torch.manual_seed(0)
    rows = {
        "doc_input_ids": [torch.randint(0, 100, (12,)).tolist() for _ in range(n)],
        "query_input_ids": [torch.randint(0, 100, (5,)).tolist() for _ in range(n)],
        "answer_input_ids": [torch.randint(0, 100, (3,)).tolist() for _ in range(n)],
        "gold_idx": [i % 4 for i in range(n)],
        "hard": [False] * n,
    }
    Dataset.from_dict(rows).save_to_disk(path)
    return path


def test_eval_smoke_writes_metrics(tmp_path, tiny_model_path):
    ds_path = _make_dataset(str(tmp_path / "ds"))
    ckpt_dir = str(tmp_path / "ckpt")
    cfg = STILLConfig(
        model_name=tiny_model_path, num_latents=4, latent_dim=16, num_blocks=2, device="cpu"
    )
    out = run_training(
        dataset_path=ds_path,
        model_name=tiny_model_path,
        cfg=cfg,
        steps=2,
        lr=1e-2,
        ckpt_dir=ckpt_dir,
        ckpt_every=0,
    )
    out_json = str(tmp_path / "metrics.json")
    metrics = run_eval(
        dataset_path=ds_path,
        ckpt_path=out["ckpt"],
        model_name=tiny_model_path,
        cfg=cfg,
        limit=4,
        out_path=out_json,
    )

    import json
    import os

    assert os.path.exists(out_json)
    saved = json.load(open(out_json))
    for key in (
        "mcq_accuracy",
        "ce_utilization",
        "compression_ratio",
        "peak_memory_compact",
        "peak_memory_full",
    ):
        assert key in saved, key
    assert 0.0 <= metrics["mcq_accuracy"] <= 1.0
    assert metrics["compression_ratio"] == 12 / 4  # avg doc len / num_latents
