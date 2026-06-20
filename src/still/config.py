from __future__ import annotations

from dataclasses import dataclass


@dataclass
class STILLConfig:
    """Configuration for the STILL basic loop.

    Shapes/symbols (see plan.md): T = doc context length, t = number of latents
    (t << T), d_lat = perceiver internal dim.
    """

    model_name: str = "Qwen/Qwen3-0.6B"
    dataset: str = "emozilla/quality"
    max_doc_tokens: int = 2048  # T: truncate article to this many tokens
    num_latents: int = 256  # t: 8x compression at T=2048
    latent_dim: int = 256  # d_lat
    num_blocks: int = 2
    lr: float = 1e-4
    steps: int = 500
    batch_size: int = 1  # one doc per step for v1 (long contexts)
    max_answer_tokens: int = 8  # on-policy letter answer is short
    seed: int = 0
    device: str = "cuda"
