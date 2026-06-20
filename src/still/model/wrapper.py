"""``STILLModel``: a frozen HF base model + the perceiver orchestration.

Three operations:

- ``compress(doc_ids)``: prefill the doc with a no-grad forward, capture per-layer
  K/V, run the perceiver (with grad) to produce a ``CompactCache``.
- ``decode(query_ids, answer_ids, cache)``: forward ``[query; answer]`` against the
  compact cache with the registered ``"still"`` attention; return answer-span logits.
- ``teacher_logits(doc_ids, query_ids, answer_ids)``: full-cache forward, no grad,
  stock attention; return answer-span logits.

Single-GPU by default. For a base too large to fit one GPU (e.g. Qwen3-32B), pass
``device_map="auto"`` to shard the frozen base across GPUs; the perceiver lives on one
device and the compact K/V + per-layer bias are placed on each layer's device so the
cross-device base forward works. Gradients flow back across devices to the perceiver.

v1 is naive: post-RoPE/post-norm K/V are compressed directly (no RoPE strip/re-apply,
no final-RMSNorm removal, no identity init).
"""

from __future__ import annotations

import torch
from torch import nn

from still.config import STILLConfig
from still.model.attention import CompactCache, register_still_attention


class STILLModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        cfg: STILLConfig | None = None,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
        device_map: str | None = None,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM

        from still.model.perceiver import STILLPerceiver

        register_still_attention()

        self.cfg = cfg or STILLConfig(model_name=model_name)
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_map = device_map

        if device_map:
            # shard the frozen base across GPUs; do not .to()
            self.base = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=dtype, attn_implementation="eager", device_map=device_map
            )
            self.input_device = self.base.get_input_embeddings().weight.device
            self.perceiver_device = torch.device("cuda:0")
        else:
            self.base = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=dtype, attn_implementation="eager"
            ).to(self.device_str)
            self.input_device = torch.device(self.device_str)
            self.perceiver_device = torch.device(self.device_str)

        self.base.requires_grad_(False)
        self.base.eval()

        self.perceiver = STILLPerceiver.from_model_config(self.base.config, self.cfg).to(
            device=self.perceiver_device, dtype=dtype
        )

        # the attention modules whose per-layer bias we attach at decode time
        self._attn_modules = [layer.self_attn for layer in self.base.model.layers]
        # device of each layer (= where its compact K/V and bias must live)
        self._layer_devices = [next(m.parameters()).device for m in self._attn_modules]

    # ------------------------------------------------------------------ helpers
    def _as_input(self, ids) -> torch.Tensor:
        if isinstance(ids, torch.Tensor):
            t = ids
        else:
            t = torch.tensor(ids, dtype=torch.long)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t.to(self.input_device)

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
            # captured K/V live on layer i's device; bring them to the perceiver's device
            key = capture.layers[i].keys[0].to(self.perceiver_device)  # [H_kv, T, D]
            value = capture.layers[i].values[0].to(self.perceiver_device)
            compact_k, compact_v, bias = self.perceiver.forward_layer(i, key, value)
            cache.add(compact_k, compact_v, bias)
        return cache

    def _set_biases(self, biases: list[torch.Tensor] | None) -> None:
        for module, layer_bias in zip(self._attn_modules, biases or [None] * len(self._attn_modules)):
            module._still_bias = layer_bias

    def _build_cache(self, cache: CompactCache):
        """DynamicCache prefilled with compact K/V, each on its layer's device."""
        from transformers import DynamicCache

        dyn = DynamicCache()
        for i in range(cache.num_layers):
            dev = self._layer_devices[i]
            k = cache.compact_k[i].unsqueeze(0).to(dev)  # [1, H_kv, t, D]
            v = cache.compact_v[i].unsqueeze(0).to(dev)
            dyn.update(k, v, i)
        return dyn

    def decode(self, query_ids, answer_ids, cache: CompactCache) -> torch.Tensor:
        """Forward [query; answer] against the compact cache; return [a_len, vocab] logits."""
        query = self._as_input(query_ids)
        answer = self._as_input(answer_ids)
        q_len, a_len = query.shape[1], answer.shape[1]
        live = torch.cat([query, answer], dim=1)

        dyn = self._build_cache(cache)
        biases = [cache.bias[i].to(self._layer_devices[i]) for i in range(cache.num_layers)]
        prev_impl = self.base.config._attn_implementation
        self._set_biases(biases)
        self.base.config._attn_implementation = "still"
        try:
            out = self.base(input_ids=live, past_key_values=dyn, use_cache=True)
        finally:
            self.base.config._attn_implementation = prev_impl
            self._set_biases(None)

        logits = out.logits[0].to(self.perceiver_device)  # [q_len + a_len, vocab]
        # positions q_len-1 .. q_len-1+a_len-1 predict answer tokens 0 .. a_len-1
        return logits[q_len - 1 : q_len - 1 + a_len]

    @torch.no_grad()
    def compact_tokens(self, token_ids) -> CompactCache:
        """Inference-time compaction of an arbitrary token span (no grad).

        Same machinery as ``compress`` but fully no-grad and fed any tokens (a chat
        prefix chunk, not just a doc). Returns one compact block (per-layer K/V + bias).
        """
        from transformers import DynamicCache

        toks = self._as_input(token_ids)
        capture = DynamicCache()
        self.base(input_ids=toks, use_cache=True, past_key_values=capture)
        cache = CompactCache()
        for i in range(len(self._attn_modules)):
            key = capture.layers[i].keys[0].to(self.perceiver_device)
            value = capture.layers[i].values[0].to(self.perceiver_device)
            ck, cv, bias = self.perceiver.forward_layer(i, key, value)
            cache.add(ck, cv, bias)
        return cache

    @torch.no_grad()
    def generate_compacted(
        self,
        token_ids,
        max_new_tokens: int = 256,
        threshold: int = 16384,
        compaction_chunk: int = 2048,
        min_live: int = 512,
        safety: int = 256,
        **gen_kwargs,
    ) -> list[int]:
        """Generate against a compacted KV cache so the attended length stays bounded.

        Compacts the oldest tokens into compact blocks until
        ``compact_len + live + max_new_tokens (+safety) <= threshold``, then runs HF
        ``generate`` with the compact blocks as the cache prefix and the recent tokens live.
        Returns only the newly generated token ids.
        """
        if isinstance(token_ids, torch.Tensor):
            tokens = token_ids.flatten().tolist()
        else:
            tokens = list(token_ids)
        acc: CompactCache | None = None
        budget = threshold - max_new_tokens - safety

        def compact_len() -> int:
            return acc.num_latents if acc is not None else 0

        while compact_len() + len(tokens) > budget and len(tokens) > min_live + compaction_chunk:
            chunk_toks = tokens[:compaction_chunk]
            tokens = tokens[compaction_chunk:]
            block = self.compact_tokens(chunk_toks)
            if acc is None:
                acc = block
            else:
                acc.extend(block)

        live = self._as_input(tokens)
        eos = self.base.config.eos_token_id
        if acc is None:
            gen = self.base.generate(
                input_ids=live, max_new_tokens=max_new_tokens, eos_token_id=eos, **gen_kwargs
            )
            return gen[0, live.shape[1] :].tolist()

        dyn = self._build_cache(acc)
        biases = [acc.bias[i].to(self._layer_devices[i]) for i in range(acc.num_layers)]
        prev_impl = self.base.config._attn_implementation
        self._set_biases(biases)
        self.base.config._attn_implementation = "still"
        try:
            gen = self.base.generate(
                input_ids=live,
                past_key_values=dyn,
                use_cache=True,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos,
                **gen_kwargs,
            )
        finally:
            self.base.config._attn_implementation = prev_impl
            self._set_biases(None)
        return gen[0, live.shape[1] :].tolist()

    @torch.no_grad()
    def teacher_logits(self, doc_ids, query_ids, answer_ids) -> torch.Tensor:
        """Full-cache forward (stock attention, no grad); return [a_len, vocab] logits."""
        doc = self._as_input(doc_ids)
        query = self._as_input(query_ids)
        answer = self._as_input(answer_ids)
        t_len, q_len, a_len = doc.shape[1], query.shape[1], answer.shape[1]
        full = torch.cat([doc, query, answer], dim=1)
        out = self.base(input_ids=full, use_cache=False)
        logits = out.logits[0].to(self.perceiver_device)
        offset = t_len + q_len - 1
        return logits[offset : offset + a_len]
