"""``still-train``: the distillation loop.

Per step: teacher logits (no grad, full cache) and student logits (grad, compact cache),
forward-KL ``KL(teacher || student)`` on the answer span only, Adam on the perceiver
params only, checkpoint. The base model stays frozen and in eval mode.

Optional Weights & Biases logging (``--wandb-project``) records the training KL curve,
perceiver grad norm, and periodic validation metrics (``--val-dataset`` + ``--eval-every``).
"""

from __future__ import annotations

import argparse
import os
import time

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
    grad_clip: float = 0.0,
    log_every: int = 1,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
    wandb_entity: str | None = None,
    val_dataset_path: str | None = None,
    eval_every: int = 0,
    eval_limit: int = 64,
) -> dict:
    from datasets import load_from_disk

    torch.manual_seed(seed)
    ds = load_from_disk(dataset_path)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    if len(ds) == 0:
        raise SystemExit(f"dataset at {dataset_path} is empty")

    val_ds = None
    if val_dataset_path:
        val_ds = load_from_disk(val_dataset_path)
        if eval_limit:
            val_ds = val_ds.select(range(min(eval_limit, len(val_ds))))

    model = STILLModel(model_name, cfg=cfg, device=cfg.device)
    model.perceiver.train()
    opt = torch.optim.Adam(model.perceiver.parameters(), lr=lr)

    # optional W&B
    wb = None
    if wandb_project:
        import wandb

        wb = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name,
            config={
                "model_name": model_name,
                "num_latents": cfg.num_latents,
                "latent_dim": cfg.latent_dim,
                "num_blocks": cfg.num_blocks,
                "max_doc_tokens": cfg.max_doc_tokens,
                "lr": lr,
                "steps": steps,
                "grad_clip": grad_clip,
                "seed": seed,
                "train_rows": len(ds),
                "val_rows": len(val_ds) if val_ds is not None else 0,
                "device": cfg.device,
            },
        )

    # tokenizer only needed for periodic eval
    tokenizer = None
    if val_ds is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)

    os.makedirs(ckpt_dir, exist_ok=True)
    losses: list[float] = []
    final_ckpt = os.path.join(ckpt_dir, "perceiver_final.pt")

    def _run_eval_and_log(step: int):
        from still.eval import evaluate_model

        metrics = evaluate_model(model, val_ds, tokenizer, cfg)
        print(
            f"  [eval @ {step}] mcq {metrics['mcq_accuracy']:.3f} "
            f"(full {metrics['mcq_accuracy_full_baseline']:.3f}) "
            f"ce_util {metrics['ce_utilization']:.3f}"
        )
        if wb is not None:
            wb.log({f"val/{k}": v for k, v in metrics.items()}, step=step)
        return metrics

    use_cuda = cfg.device.startswith("cuda") and torch.cuda.is_available()

    # baseline validation on the untrained (random-init) perceiver
    if eval_every and val_ds is not None:
        _run_eval_and_log(0)

    for step in range(steps):
        t0 = time.perf_counter()
        row = ds[step % len(ds)]
        doc, query, answer = _row_to_tensors(row, model.device_str)

        with torch.no_grad():
            t_logits = model.teacher_logits(doc, query, answer)
        cache = model.compress(doc)
        s_logits = model.decode(query, answer, cache)
        loss = kl_teacher_student(s_logits, t_logits)

        opt.zero_grad()
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.perceiver.parameters(), grad_clip)
        gnorm = perceiver_grad_norm(model)
        opt.step()

        # extra per-step diagnostics
        answer_targets = answer.squeeze(0)
        teacher_ce = F.cross_entropy(t_logits.float(), answer_targets).item()
        student_ce = F.cross_entropy(s_logits.detach().float(), answer_targets).item()
        step_time = time.perf_counter() - t0

        losses.append(loss.item())
        if log_every and step % log_every == 0:
            print(f"step {step:5d} | kl_nats/tok {loss.item():.4f} | perceiver_grad_norm {gnorm:.4f}")
        if wb is not None:
            log = {
                "train/loss": loss.item(),
                "train/kl_nats_per_token": loss.item(),
                "train/perceiver_grad_norm": gnorm,
                "train/lr": opt.param_groups[0]["lr"],
                "train/student_answer_ce": student_ce,
                "train/teacher_answer_ce": teacher_ce,
                "train/student_ce_gap": student_ce - teacher_ce,
                "train/step_time_s": step_time,
                "train/doc_len": int(doc.shape[1]),
                "train/answer_len": int(answer.shape[1]),
            }
            if use_cuda:
                log["train/gpu_mem_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
            wb.log(log, step=step)

        if eval_every and val_ds is not None and (step + 1) % eval_every == 0:
            _run_eval_and_log(step + 1)

        if ckpt_every and (step + 1) % ckpt_every == 0:
            path = os.path.join(ckpt_dir, f"perceiver_step_{step + 1}.pt")
            torch.save(model.perceiver.state_dict(), path)

    torch.save(model.perceiver.state_dict(), final_ckpt)
    print(f"wrote checkpoint to {final_ckpt}")

    final_metrics = None
    if val_ds is not None:
        final_metrics = _run_eval_and_log(steps)
    if wb is not None:
        if final_metrics:
            wb.summary.update({f"final/{k}": v for k, v in final_metrics.items()})
        wb.finish()

    return {"losses": losses, "ckpt": final_ckpt, "final_metrics": final_metrics}


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
    ap.add_argument("--grad-clip", type=float, default=0.0)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--num-latents", type=int, default=256)
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--num-blocks", type=int, default=2)
    ap.add_argument("--max-doc-tokens", type=int, default=2048)
    ap.add_argument("--device", default=None)
    # W&B + periodic eval
    ap.add_argument("--wandb-project", default=None, help="enable W&B logging to this project")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--val-dataset", default=None, help="held-out cache for periodic eval")
    ap.add_argument("--eval-every", type=int, default=0, help="run val eval every N steps (0=off)")
    ap.add_argument("--eval-limit", type=int, default=64)
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
        grad_clip=args.grad_clip,
        log_every=args.log_every,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_entity=args.wandb_entity,
        val_dataset_path=args.val_dataset,
        eval_every=args.eval_every,
        eval_limit=args.eval_limit,
    )


if __name__ == "__main__":
    main()
