from __future__ import annotations

from still.data.generate_answers import tokenize_doc_query
from still.data.quality import ANSWER_CUE, LETTERS, format_query, letter_token_ids


def test_format_query_renders_four_options_and_cue(synthetic_row):
    q = format_query(synthetic_row["question"], synthetic_row["options"])
    for i, opt in enumerate(synthetic_row["options"]):
        assert f"{LETTERS[i]}. {opt}" in q
    assert q.endswith(ANSWER_CUE)
    assert synthetic_row["question"] in q


def test_tokenize_doc_query_nonempty_spans(tokenizer, synthetic_row):
    doc_ids, query_ids = tokenize_doc_query(
        tokenizer,
        synthetic_row["article"],
        synthetic_row["question"],
        synthetic_row["options"],
        max_doc_tokens=32,
    )
    assert len(doc_ids) > 0
    assert len(query_ids) > 0
    # Doc truncation is honored.
    assert len(doc_ids) <= 32


def test_letter_token_ids(tokenizer):
    ids = letter_token_ids(tokenizer)
    assert len(ids) == 4
    assert all(isinstance(i, int) for i in ids)
