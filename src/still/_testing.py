"""Helpers for building a tiny, deterministic Qwen3 model for tests and smoke runs.

Kept inside the shipped package so both the pytest fixtures and the shell-level
smoke commands in the plan can produce an identical tiny model offline.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

TINY_TOKENIZER_SOURCE = "Qwen/Qwen3-0.6B"


def build_tiny_model(seed: int = 0) -> Qwen3ForCausalLM:
    """A tiny random-weight Qwen3 with GQA (H_q=4, H_kv=2) for fast CPU tests."""
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(TINY_TOKENIZER_SOURCE)
    cfg = Qwen3Config(
        vocab_size=tok.vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    model = Qwen3ForCausalLM(cfg)
    model.eval()
    return model


def make_tiny_model(out_dir: str, seed: int = 0) -> str:
    """Build a tiny Qwen3 + the real Qwen tokenizer and save both to ``out_dir``.

    Returns ``out_dir`` so callers can pass it straight to ``--model``.
    """
    model = build_tiny_model(seed=seed)
    tok = AutoTokenizer.from_pretrained(TINY_TOKENIZER_SOURCE)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Save a tiny Qwen3 model + tokenizer to a directory.")
    ap.add_argument("out_dir")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    path = make_tiny_model(args.out_dir, seed=args.seed)
    print(f"wrote tiny model to {path}")


if __name__ == "__main__":
    main()
