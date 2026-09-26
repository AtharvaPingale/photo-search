"""Contrastive fine-tuning of OpenCLIP on this library's (image, text) pairs.

Two modes:
  last-layers  unfreeze the last N transformer blocks of both towers (+ final norms,
               projections and the temperature); everything else stays frozen.
  lora         low-rank adapters on every block's attention (in/out projections) and
               MLP weights, via torch parametrizations so they also work on
               nn.MultiheadAttention's fused in_proj_weight. Base weights frozen;
               adapters are merged into the weights when saving.

Loss: symmetric InfoNCE (the CLIP loss). Pairs in a batch that share the exact same
text (common with keyword templates) are masked out of each other's negatives.

Robustness: `--wise-alpha` saves WiSE-FT weights, alpha * finetuned + (1 - alpha) * base.
Interpolating back toward the base model usually keeps most of the in-domain gain
while giving back less of CLIP's general knowledge; pick alpha on the eval *dev* split.

The result is saved as a new image model (data/models/<name>/) so `photo-search embed
--model <name>` indexes it next to the base embeddings and `eval.ablation --models`
compares them per category.

    uv run python -m training.build_pairs
    uv run python -m training.finetune_clip --mode lora --name finetuned-v1
    uv run photo-search embed --model finetuned-v1
    uv run python -m eval.ablation --models openclip-vitb32,finetuned-v1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, Dataset

from api.config import get_settings
from api.ml.registry import get_image_model_spec, save_finetuned_spec


class PairDataset(Dataset):
    def __init__(self, path: Path, transform, tokenizer):
        self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        with Image.open(r["image"]) as im:
            img = self.transform(im.convert("RGB"))
        return img, self.tokenizer([r["text"]])[0], r["text"]


def collate(batch):
    imgs, toks, texts = zip(*batch, strict=True)
    return torch.stack(imgs), torch.stack(toks), list(texts)


# ------------------------------------------------------------------ LoRA


class LoRA(nn.Module):
    """W + (alpha / r) * B @ A, applied through torch.nn.utils.parametrize."""

    def __init__(self, out_features: int, in_features: int, rank: int, alpha: float):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(
            torch.zeros(out_features, rank)
        )  # zero init: starts as the base model
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        return W + self.scale * (self.B @ self.A).to(W.dtype)


def add_lora(model: Any, rank: int = 8, alpha: float = 16.0) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for p in model.parameters():
        p.requires_grad_(False)
    for name, mod in model.named_modules():
        targets: list[tuple[nn.Module, str]] = []
        if isinstance(mod, nn.MultiheadAttention) and mod.in_proj_weight is not None:
            targets.append((mod, "in_proj_weight"))
            targets.append((mod.out_proj, "weight"))
        elif isinstance(mod, nn.Linear) and name.split(".")[-1] in ("c_fc", "c_proj"):
            targets.append((mod, "weight"))
        for m, attr in targets:
            W = getattr(m, attr)
            lora = LoRA(W.shape[0], W.shape[1], rank, alpha).to(W.device)
            parametrize.register_parametrization(m, attr, lora)
            params += [lora.A, lora.B]
    return params


def merge_lora(model: Any) -> None:
    for mod in list(model.modules()):
        if parametrize.is_parametrized(mod):
            for attr in list(mod.parametrizations.keys()):
                parametrize.remove_parametrizations(mod, attr, leave_parametrized=True)


def unfreeze_last(model: Any, n_blocks: int) -> list[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad_(False)
    trainable: list[nn.Parameter] = []
    for tower in (model.visual.transformer.resblocks, model.transformer.resblocks):
        for blk in list(tower)[-n_blocks:]:
            for p in blk.parameters():
                p.requires_grad_(True)
                trainable.append(p)
    for mod in (model.visual.ln_post, model.ln_final):
        for p in mod.parameters():
            p.requires_grad_(True)
            trainable.append(p)
    for p in (model.visual.proj, model.text_projection, model.logit_scale):
        if isinstance(p, nn.Parameter):
            p.requires_grad_(True)
            trainable.append(p)
    return trainable


# ------------------------------------------------------------------ training


def clip_loss(
    img: torch.Tensor, txt: torch.Tensor, logit_scale: torch.Tensor, texts: list[str]
) -> torch.Tensor:
    img = F.normalize(img, dim=-1)
    txt = F.normalize(txt, dim=-1)
    logits = logit_scale.exp().clamp(max=100) * img @ txt.T
    # identical texts in one batch are not negatives of each other
    same = torch.tensor([[a == b for b in texts] for a in texts], device=logits.device)
    same.fill_diagonal_(False)
    logits = logits.masked_fill(same, float("-inf"))
    labels = torch.arange(len(texts), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    """Text->image recall@1/5 within the validation set (one pair per image is enough)."""
    model.eval()
    img_vecs, txt_vecs, keys = [], [], []
    for imgs, toks, texts in loader:
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            img_vecs.append(F.normalize(model.encode_image(imgs.to(device)).float(), dim=-1))
            txt_vecs.append(F.normalize(model.encode_text(toks.to(device)).float(), dim=-1))
        keys += texts
    if not img_vecs:
        return {}
    Im, Tm = torch.cat(img_vecs), torch.cat(txt_vecs)
    sims = Tm @ Im.T
    ranks = (sims > sims.diag().unsqueeze(1)).sum(1)
    model.train()
    return {
        "val_r@1": (ranks < 1).float().mean().item(),
        "val_r@5": (ranks < 5).float().mean().item(),
    }


def train(a: argparse.Namespace) -> dict[str, Any]:
    import open_clip

    from eval.tracking import git_commit, mlflow_run

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = get_image_model_spec(a.base)
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        base.arch, pretrained=base.pretrained
    )
    if base.checkpoint:
        model.load_state_dict(torch.load(base.checkpoint, map_location="cpu", weights_only=True))
    base_state = (
        {k: v.detach().clone() for k, v in model.state_dict().items()} if a.wise_alpha < 1 else None
    )
    model = model.to(device)
    tok = open_clip.get_tokenizer(base.arch)
    tr = PairDataset(a.data / "train.jsonl", preprocess_train, tok)
    va = PairDataset(a.data / "val.jsonl", preprocess_val, tok)
    if len(tr) < a.batch_size:
        raise SystemExit(f"only {len(tr)} training pairs; caption more photos first")
    dl = DataLoader(
        tr,
        batch_size=a.batch_size,
        shuffle=True,
        num_workers=a.workers,
        collate_fn=collate,
        drop_last=True,
    )
    vl = DataLoader(va, batch_size=256, shuffle=False, num_workers=a.workers, collate_fn=collate)

    params = (
        add_lora(model, a.rank, a.lora_alpha)
        if a.mode == "lora"
        else unfreeze_last(model, a.unfreeze)
    )
    if a.mode == "lora":
        model.logit_scale.requires_grad_(True)
        params.append(model.logit_scale)
    n_train = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    total = a.epochs * len(dl)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))),
    )
    history = []
    run_name = f"{a.name}-{a.mode}"
    with mlflow_run("finetune", run_name, enabled=not a.no_mlflow) as run:
        run.log_params({**{k: str(v) for k, v in vars(a).items()}, "trainable_params": n_train, "git_commit": git_commit(),
                        "n_train_pairs": len(tr), "n_val_pairs": len(va)})  # fmt: skip
        before = evaluate(model, vl, device)
        run.log_metrics({f"{k}_before": v for k, v in before.items()})
        print(f"trainable params: {n_train:,}  before: {before}")
        step = 0
        model.train()
        for ep in range(a.epochs):
            t0, losses = time.time(), []
            for imgs, toks, texts in dl:
                with torch.autocast(
                    device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"
                ):
                    loss = clip_loss(
                        model.encode_image(imgs.to(device)).float(),
                        model.encode_text(toks.to(device)).float(),
                        model.logit_scale,
                        texts,
                    )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                losses.append(loss.item())
                step += 1
            m = {
                "epoch": ep,
                "loss": float(np.mean(losses)),
                **evaluate(model, vl, device),
                "epoch_s": time.time() - t0,
            }
            history.append(m)
            run.log_metrics({k: v for k, v in m.items() if k != "epoch"})
            print(m)
        if a.mode == "lora":
            merge_lora(model)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if base_state is not None:
            state = {
                k: a.wise_alpha * v.float() + (1 - a.wise_alpha) * base_state[k].float()
                for k, v in state.items()
            }
        out_dir = get_settings().models_dir / a.name
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, out_dir / "model.pt")
        save_finetuned_spec(a.name, base, "model.pt")
        summary = {"name": a.name, "base": base.name, "mode": a.mode, "trainable_params": n_train,
                   "before": before, "history": history, "wise_alpha": a.wise_alpha,
                   "mlflow_run_id": run.run_id}  # fmt: skip
        (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", default="finetuned-v1")
    ap.add_argument("--base", default=None, help="base image model (default PS_IMAGE_MODEL)")
    ap.add_argument("--data", type=Path, default=get_settings().data_dir / "training")
    ap.add_argument("--mode", choices=["lora", "last-layers"], default="lora")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--unfreeze", type=int, default=2, help="last-layers mode: blocks per tower")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--wise-alpha", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-mlflow", action="store_true")
    a = ap.parse_args()
    if a.lr is None:
        a.lr = 1e-4 if a.mode == "lora" else 1e-5
    print(json.dumps(train(a), indent=2, default=str))


if __name__ == "__main__":
    main()
