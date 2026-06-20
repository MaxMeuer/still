"""``still-train``: the distillation loop.

Per step: teacher logits (no grad, full cache) and student logits (grad, compact cache),
forward-KL ``KL(teacher || student)`` on the answer span only, Adam on the perceiver
params only, checkpoint. The base model stays frozen and in eval mode.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F

from still.config import STILLConfig
from still.model.wrapper import STILLModel


def _row_to_tensors(row, device):
    doc = torch.tensor([row["doc_input_ids"]], dtype=torch.long, device=device)
    query = torch.tensor([row["query_input_ids"]], dtype=torch.long, device=device)
    answer = torch.tensor([row["answer_input_ids"]], dtype=torch.long, device=device)
    return doc, query, answer


def kl_teacher_student(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """Forward KL(teacher || student) over the answer span, in nats/token."""
    log_student = F.log_softmax(student_logits.float(), dim=-1)
    teacher_probs = F.softmax(teacher_logits.float(), dim=-1)
    # F.kl_div(input=log q, target=p) == sum p * (log p - log q) == KL(p || q); batchmean -> per token
    return F.kl_div(log_student, teacher_probs, reduction="batchmean")


def perceiver_grad_norm(model: STILLModel) -> float:
    total = 0.0
    for p in model.perceiver.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total**0.5


def run_training(
    *,
    dataset_path: str,
    model_name: str,
    cfg: STILLConfig,
    steps: int,
    lr: float,
    ckpt_dir: str,
    ckpt_every: int = 100,
    limit: int | None = None,
    seed: int = 0,
) -> dict:
    from datasets import load_from_disk

    torch.manual_seed(seed)
    ds = load_from_disk(dataset_path)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    if len(ds) == 0:
        raise SystemExit(f"dataset at {dataset_path} is empty")

    model = STILLModel(model_name, cfg=cfg, device=cfg.device)
    model.perceiver.train()
    opt = torch.optim.Adam(model.perceiver.parameters(), lr=lr)

    os.makedirs(ckpt_dir, exist_ok=True)
    losses: list[float] = []
    final_ckpt = os.path.join(ckpt_dir, "perceiver_final.pt")

    for step in range(steps):
        row = ds[step % len(ds)]
        doc, query, answer = _row_to_tensors(row, model.device_str)

        with torch.no_grad():
            t_logits = model.teacher_logits(doc, query, answer)
        cache = model.compress(doc)
        s_logits = model.decode(query, answer, cache)
        loss = kl_teacher_student(s_logits, t_logits)

        opt.zero_grad()
        loss.backward()
        gnorm = perceiver_grad_norm(model)
        opt.step()

        losses.append(loss.item())
        print(f"step {step:5d} | kl_nats/tok {loss.item():.4f} | perceiver_grad_norm {gnorm:.4f}")

        if ckpt_every and (step + 1) % ckpt_every == 0:
            path = os.path.join(ckpt_dir, f"perceiver_step_{step + 1}.pt")
            torch.save(model.perceiver.state_dict(), path)

    torch.save(model.perceiver.state_dict(), final_ckpt)
    print(f"wrote checkpoint to {final_ckpt}")
    return {"losses": losses, "ckpt": final_ckpt}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train the STILL perceiver by forward-KL distillation.")
    ap.add_argument("--dataset", required=True, help="path produced by still-preprocess")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
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
        lr=args.lr,
        steps=args.steps,
        seed=args.seed,
        device=device,
    )
    run_training(
        dataset_path=args.dataset,
        model_name=args.model,
        cfg=cfg,
        steps=args.steps,
        lr=args.lr,
        ckpt_dir=args.ckpt_dir,
        ckpt_every=args.ckpt_every,
        limit=args.limit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
