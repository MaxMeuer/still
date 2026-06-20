"""``still-eval``: measure the basic loop against the full-cache baseline.

Reports four metric families on a held-out preprocessed split:
- MCQ accuracy (compact cache vs full-cache baseline),
- CE utilization (``CE_full / CE_compact`` on the answer span; closer to 1.0 is better),
- achieved compression (``T / t`` and the compact-vs-full KV memory ratio),
- peak memory (compact decode vs full decode).
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from still.config import STILLConfig
from still.data.quality import letter_token_ids
from still.model.wrapper import STILLModel


def _answer_ce(logits: torch.Tensor, answer_ids: list[int]) -> float:
    targets = torch.tensor(answer_ids, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits.float(), targets).item()


def _kv_memory_ratio(num_layers, num_kv_heads, head_dim, doc_len, num_latents) -> float:
    # full: K and V over doc_len; compact: K and V over t, plus one bias scalar per latent
    full = num_layers * num_kv_heads * doc_len * head_dim * 2
    compact = num_layers * num_kv_heads * num_latents * head_dim * 2 + num_layers * num_kv_heads * num_latents
    return compact / full


def _measured(fn, device: str):
    """Run ``fn`` and return ``(result, peak_bytes)`` (peak is 0 off-CUDA)."""
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        result = fn()
        torch.cuda.synchronize()
        return result, int(torch.cuda.max_memory_allocated())
    return fn(), 0


def evaluate_model(model: STILLModel, ds, tokenizer, cfg: STILLConfig) -> dict:
    """Compute the eval metrics for an already-built model over a dataset.

    Reused by both ``run_eval`` (loads a checkpoint) and the training loop (periodic
    validation). Restores the perceiver's train/eval mode on exit.
    """
    letters = letter_token_ids(tokenizer)
    letter_idx = torch.tensor(letters, device=cfg.device)
    was_training = model.perceiver.training
    model.perceiver.eval()

    compact_correct = full_correct = total = 0
    ce_compact_sum = ce_full_sum = 0.0
    doc_len_sum = 0
    peak_compact = peak_full = 0

    try:
        with torch.no_grad():
            for row in ds:
                doc = torch.tensor([row["doc_input_ids"]], dtype=torch.long, device=cfg.device)
                query = torch.tensor([row["query_input_ids"]], dtype=torch.long, device=cfg.device)
                answer = torch.tensor([row["answer_input_ids"]], dtype=torch.long, device=cfg.device)
                gold = int(row["gold_idx"])
                doc_len_sum += doc.shape[1]

                # --- compact path (compress + decode against the compact cache)
                def _compact(doc=doc, query=query, answer=answer):
                    cache = model.compress(doc)
                    return model.decode(query, answer, cache)

                s_logits, peak = _measured(_compact, cfg.device)
                peak_compact = max(peak_compact, peak)

                # --- full path (baseline)
                def _full(doc=doc, query=query, answer=answer):
                    return model.teacher_logits(doc, query, answer)

                t_logits, peak = _measured(_full, cfg.device)
                peak_full = max(peak_full, peak)

                # MCQ: row 0 of the answer-predicting logits scores the letter token
                compact_pred = int(s_logits[0].index_select(0, letter_idx).argmax().item())
                full_pred = int(t_logits[0].index_select(0, letter_idx).argmax().item())
                compact_correct += int(compact_pred == gold)
                full_correct += int(full_pred == gold)

                ce_compact_sum += _answer_ce(s_logits, row["answer_input_ids"])
                ce_full_sum += _answer_ce(t_logits, row["answer_input_ids"])
                total += 1
    finally:
        model.perceiver.train(was_training)

    ce_compact = ce_compact_sum / max(total, 1)
    ce_full = ce_full_sum / max(total, 1)
    avg_doc_len = doc_len_sum / max(total, 1)
    mc = model.base.config
    head_dim = getattr(mc, "head_dim", mc.hidden_size // mc.num_attention_heads)

    return {
        "n": total,
        "mcq_accuracy": compact_correct / max(total, 1),
        "mcq_accuracy_full_baseline": full_correct / max(total, 1),
        "ce_compact": ce_compact,
        "ce_full": ce_full,
        "ce_utilization": (ce_full / ce_compact) if ce_compact > 0 else float("nan"),
        "compression_ratio": avg_doc_len / cfg.num_latents,
        "kv_memory_ratio": _kv_memory_ratio(
            mc.num_hidden_layers, mc.num_key_value_heads, head_dim, avg_doc_len, cfg.num_latents
        ),
        "peak_memory_compact": peak_compact,
        "peak_memory_full": peak_full,
    }


def run_eval(
    *,
    dataset_path: str,
    ckpt_path: str,
    model_name: str,
    cfg: STILLConfig,
    limit: int | None = None,
    out_path: str | None = None,
) -> dict:
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    ds = load_from_disk(dataset_path)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    model = STILLModel(model_name, cfg=cfg, device=cfg.device)
    state = torch.load(ckpt_path, map_location=cfg.device)
    model.perceiver.load_state_dict(state)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    metrics = evaluate_model(model, ds, tokenizer, cfg)

    print(json.dumps(metrics, indent=2))
    if out_path:
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"wrote metrics to {out_path}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate a trained STILL perceiver.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", required=True, help="perceiver checkpoint (.pt)")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    ap.add_argument("--num-latents", type=int, default=256)
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--num-blocks", type=int, default=2)
    ap.add_argument("--max-doc-tokens", type=int, default=2048)
    ap.add_argument("--device", default=None)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = STILLConfig(
        model_name=args.model,
        num_latents=args.num_latents,
        latent_dim=args.latent_dim,
        num_blocks=args.num_blocks,
        max_doc_tokens=args.max_doc_tokens,
        device=device,
    )
    run_eval(
        dataset_path=args.dataset,
        ckpt_path=args.ckpt,
        model_name=args.model,
        cfg=cfg,
        limit=args.limit,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
