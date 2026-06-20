"""``STILLModel``: a frozen HF base model + the perceiver orchestration.

Three operations:

- ``compress(doc_ids)``: prefill the doc with a no-grad forward, capture per-layer
  K/V, run the perceiver (with grad) to produce a ``CompactCache``.
- ``decode(query_ids, answer_ids, cache)``: forward ``[query; answer]`` against the
  compact cache with the registered ``"still"`` attention; return answer-span logits.
- ``teacher_logits(doc_ids, query_ids, answer_ids)``: full-cache forward, no grad,
  stock attention; return answer-span logits.

v1 is naive: post-RoPE/post-norm K/V are compressed directly (no RoPE strip/re-apply,
no final-RMSNorm removal, no identity init).
"""

from __future__ import annotations

import torch
from torch import nn

from still.config import STILLConfig
from still.model.attention import CompactCache, register_still_attention
from still.model.perceiver import STILLPerceiver


class STILLModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        cfg: STILLConfig | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM

        register_still_attention()

        self.cfg = cfg or STILLConfig(model_name=model_name)
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load frozen base with eager attention (the "still" impl is swapped in per decode).
        self.base = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, attn_implementation="eager"
        ).to(self.device_str)
        self.base.requires_grad_(False)
        self.base.eval()

        self.perceiver = STILLPerceiver.from_model_config(self.base.config, self.cfg).to(
            device=self.device_str, dtype=dtype
        )

        # the attention modules whose per-layer bias we attach at decode time
        self._attn_modules = [layer.self_attn for layer in self.base.model.layers]

    # ------------------------------------------------------------------ helpers
    def _as_input(self, ids) -> torch.Tensor:
        if isinstance(ids, torch.Tensor):
            t = ids
        else:
            t = torch.tensor(ids, dtype=torch.long)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t.to(self.device_str)

    # --------------------------------------------------------------- operations
    def compress(self, doc_ids) -> CompactCache:
        """Prefill the doc, capture per-layer K/V, compress via the perceiver (grad)."""
        from transformers import DynamicCache

        doc = self._as_input(doc_ids)
        with torch.no_grad():
            capture = DynamicCache()
            self.base(input_ids=doc, use_cache=True, past_key_values=capture)

        cache = CompactCache()
        for i in range(len(self._attn_modules)):
            key = capture.layers[i].keys[0]  # [H_kv, T, D]
            value = capture.layers[i].values[0]
            compact_k, compact_v, bias = self.perceiver.forward_layer(i, key, value)
            cache.add(compact_k, compact_v, bias)
        return cache

    def _set_biases(self, biases: list[torch.Tensor] | None) -> None:
        for module, layer_bias in zip(self._attn_modules, biases or [None] * len(self._attn_modules)):
            module._still_bias = layer_bias

    def decode(self, query_ids, answer_ids, cache: CompactCache) -> torch.Tensor:
        """Forward [query; answer] against the compact cache; return [a_len, vocab] logits."""
        query = self._as_input(query_ids)
        answer = self._as_input(answer_ids)
        q_len, a_len = query.shape[1], answer.shape[1]
        live = torch.cat([query, answer], dim=1)

        dyn = cache.to_dynamic_cache()
        prev_impl = self.base.config._attn_implementation
        self._set_biases(cache.bias)
        self.base.config._attn_implementation = "still"
        try:
            out = self.base(input_ids=live, past_key_values=dyn, use_cache=True)
        finally:
            self.base.config._attn_implementation = prev_impl
            self._set_biases(None)

        logits = out.logits[0]  # [q_len + a_len, vocab]
        # positions q_len-1 .. q_len-1+a_len-1 predict answer tokens 0 .. a_len-1
        return logits[q_len - 1 : q_len - 1 + a_len]

    @torch.no_grad()
    def teacher_logits(self, doc_ids, query_ids, answer_ids) -> torch.Tensor:
        """Full-cache forward (stock attention, no grad); return [a_len, vocab] logits."""
        doc = self._as_input(doc_ids)
        query = self._as_input(query_ids)
        answer = self._as_input(answer_ids)
        t_len, q_len, a_len = doc.shape[1], query.shape[1], answer.shape[1]
        full = torch.cat([doc, query, answer], dim=1)
        out = self.base(input_ids=full, use_cache=False)
        logits = out.logits[0]
        offset = t_len + q_len - 1
        return logits[offset : offset + a_len]
