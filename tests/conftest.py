from __future__ import annotations

import pytest
import torch
from transformers import AutoTokenizer

from still._testing import TINY_TOKENIZER_SOURCE, build_tiny_model, make_tiny_model


@pytest.fixture(scope="session")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(TINY_TOKENIZER_SOURCE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="session")
def tiny_model():
    torch.manual_seed(0)
    return build_tiny_model(seed=0)


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory):
    out = tmp_path_factory.mktemp("tiny-model")
    return make_tiny_model(str(out), seed=0)


@pytest.fixture
def synthetic_row():
    return {
        "article": "The lighthouse keeper watched the storm roll in over the gray sea. "
        * 8,
        "question": "What did the keeper watch?",
        "options": ["A storm", "A ship", "A bird", "The sunrise"],
        "answer": 0,
        "hard": False,
    }
