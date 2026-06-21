"""``still-serve``: an OpenAI-compatible HTTP server backed by STILL compaction.

Serves ``/v1/models`` and ``/v1/chat/completions`` from a single GPU-resident
``STILLModel`` + trained perceiver. Each chat request is tokenized via the chat template;
if the prompt is long, the oldest tokens are compacted into the STILL compact cache so the
base model's attended length stays bounded — an agent can run arbitrarily long rollouts
without ever exceeding the context window (no more context-overflow 400s).

Pure torch/transformers + Python stdlib. Requests are serialized with a lock (one model).
Point an OpenAI client (e.g. terminus-2) at ``http://<host>:<port>/v1``.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from still.config import STILLConfig
from still.model.wrapper import STILLModel

# module-level state shared by the request handler
_STATE: dict = {}
_LOCK = threading.Lock()


def _build_state(args) -> dict:
    from transformers import AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = STILLConfig(
        model_name=args.model,
        num_latents=args.num_latents,
        latent_dim=args.latent_dim,
        num_blocks=args.num_blocks,
        device=device,
    )
    model = STILLModel(
        args.model,
        cfg=cfg,
        device=device,
        dtype=getattr(torch, args.dtype),
        device_map=args.device_map,
        attn_implementation="sdpa",  # memory-efficient prefill for long contexts
    )
    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        model.perceiver.load_state_dict(state)
        print(f"loaded perceiver checkpoint: {args.ckpt}")
    model.perceiver.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "served_name": args.served_model_name,
        "threshold": args.threshold,
        "compaction_chunk": args.compaction_chunk,
        "live_window": args.live_window,
        "default_max_new_tokens": args.max_new_tokens,
        "enable_thinking": args.enable_thinking,
    }


def _render_prompt(tokenizer, messages, enable_thinking) -> list[int]:
    """Render messages to a flat list[int] of token ids (robust across transformers versions)."""
    # Render to text first (apply_chat_template(tokenize=True) returns a BatchEncoding in
    # transformers 5.x, which is awkward to normalize); then encode to a clean list[int].
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(text, add_special_tokens=False)


def _gen_kwargs(body) -> dict:
    temperature = body.get("temperature", None)
    top_p = body.get("top_p", None)
    top_k = body.get("top_k", None)
    out: dict = {}
    if temperature is not None and float(temperature) > 0:
        out["do_sample"] = True
        out["temperature"] = float(temperature)
        if top_p is not None:
            out["top_p"] = float(top_p)
        if top_k is not None:
            out["top_k"] = int(top_k)
    elif temperature is not None and float(temperature) == 0:
        out["do_sample"] = False
    return out


def _chat_completion(body: dict) -> dict:
    st = _STATE
    model, tok = st["model"], st["tokenizer"]
    messages = body["messages"]
    max_new = int(body.get("max_tokens") or st["default_max_new_tokens"])

    prompt_ids = _render_prompt(tok, messages, st["enable_thinking"])
    with _LOCK:
        out_ids = model.generate_compacted(
            prompt_ids,
            max_new_tokens=max_new,
            threshold=st["threshold"],
            compaction_chunk=st["compaction_chunk"],
            live_window=st["live_window"],
            **_gen_kwargs(body),
        )
    text = tok.decode(out_ids, skip_special_tokens=True)
    finish = "length" if len(out_ids) >= max_new else "stop"

    return {
        "id": f"chatcmpl-still-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": st["served_name"],
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(out_ids),
            "total_tokens": len(prompt_ids) + len(out_ids),
        },
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quieter logs
        pass

    def _send(self, code: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            self._send(
                200,
                {"object": "list", "data": [{"id": _STATE["served_name"], "object": "model"}]},
            )
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/v1/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        try:
            self._send(200, _chat_completion(body))
        except Exception as e:  # noqa: BLE001 - return an OpenAI-style error
            self._send(500, {"error": {"message": str(e), "type": "internal_error"}})


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="OpenAI-compatible server with STILL KV compaction.")
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--ckpt", default=None, help="perceiver checkpoint (.pt)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--served-model-name", default="still-compact")
    ap.add_argument("--threshold", type=int, default=16384, help="prompt length that triggers compaction")
    ap.add_argument("--compaction-chunk", type=int, default=2048)
    ap.add_argument("--live-window", type=int, default=2048, help="recent tokens kept live after compaction")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--num-latents", type=int, default=256)
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--num-blocks", type=int, default=2)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--device-map", default=None, help="e.g. 'auto' to shard a large base across GPUs")
    ap.add_argument(
        "--enable-thinking",
        action="store_true",
        help="enable Qwen3 thinking (default off for speed + shorter context)",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    global _STATE
    args = build_parser().parse_args(argv)
    _STATE = _build_state(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"still-serve on {args.host}:{args.port} | model={args.model} "
        f"served_as={args.served_model_name} threshold={args.threshold} "
        f"chunk={args.compaction_chunk} thinking={args.enable_thinking}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
