"""Training pipeline with early stopping, checkpointing, and full metric logging."""

from __future__ import annotations

import logging
import os
import random
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import PAD_IDX
from src.utils.helpers import decode_sequence
from src.utils.metrics import batch_metrics

logger = logging.getLogger("VQA")


class EarlyStopping:
    """Stop training when the monitored metric stops improving.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum change to qualify as improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter: int = 0
        self.best_score: Optional[float] = None

    def __call__(self, score: float) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_model(
    model: nn.Module,
    name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    answer_vocab: Any,
    device: torch.device,
    epochs: int = 20,
    lr: float = 3e-4,
    use_beam: bool = False,
    beam_w: int = 5,
    ckpt_dir: str = "checkpoints",
    label_smoothing: float = 0.1,
    patience: int = 5,
    grad_clip: float = 5.0,
) -> dict[str, list[float]]:
    """Full training loop with Label Smoothing, Cosine Annealing, and Early Stopping.

    Args:
        model: VQAModel instance.
        name: Model variant name (e.g. "M1_Scratch_NoAttn").
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        answer_vocab: Answer Vocabulary for decoding.
        device: Torch device.
        epochs: Maximum number of training epochs.
        lr: Initial learning rate.
        use_beam: Use beam search for validation.
        beam_w: Beam width.
        ckpt_dir: Directory for saving checkpoints.
        label_smoothing: Label smoothing factor.
        patience: Early stopping patience.
        grad_clip: Maximum gradient norm.

    Returns:
        History dict with all training metrics per epoch.
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"  TRAINING: {name}  epochs={epochs}  lr={lr}")
    logger.info(f"  label_smoothing={label_smoothing}  patience={patience}  grad_clip={grad_clip}")
    logger.info(f"{'=' * 70}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    stopper = EarlyStopping(patience=patience)

    best_f1: float = 0.0
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "lr": [],
        "val_acc": [], "val_em": [], "val_f1": [], "val_meteor": [],
        "val_bleu1": [], "val_bleu2": [], "val_bleu3": [], "val_bleu4": [],
    }

    for epoch in range(1, epochs + 1):
        tf = max(0.5, 1.0 - (epoch - 1) / epochs * 0.5)

        # ── TRAIN ──
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{name}] Epoch {epoch}/{epochs} tf={tf:.2f}")

        for imgs, qs, ql, ans, al, _ in pbar:
            imgs = imgs.to(device)
            qs = qs.to(device)
            ql = ql.to(device)
            ans = ans.to(device)

            optimizer.zero_grad()
            out = model(imgs, qs, ql, ans, tf_ratio=tf)
            out = out.reshape(-1, out.size(-1))
            tgt = ans[:, 1:].reshape(-1)
            loss = criterion(out, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / len(train_loader)

        # ── VALIDATE ──
        model.eval()
        val_loss_sum = 0.0
        preds_all, refs_all = [], []

        with torch.no_grad():
            for imgs, qs, ql, ans, al, ans_txt in val_loader:
                imgs = imgs.to(device)
                qs = qs.to(device)
                ql = ql.to(device)
                ans = ans.to(device)

                # Validation loss (teacher forcing = 0)
                out = model(imgs, qs, ql, ans, tf_ratio=0)
                out_flat = out.reshape(-1, out.size(-1))
                tgt = ans[:, 1:].reshape(-1)
                val_loss_sum += criterion(out_flat, tgt).item()

                # Generation
                gen = model.generate(imgs, qs, ql, use_beam=use_beam, beam_width=beam_w)
                for i in range(gen.size(0)):
                    preds_all.append(decode_sequence(gen[i].cpu().tolist(), answer_vocab))
                    refs_all.append(ans_txt[i])

        val_loss = val_loss_sum / len(val_loader)
        m = batch_metrics(preds_all, refs_all)
        cur_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(cur_lr)
        for k in ["val_acc", "val_em", "val_f1", "val_meteor",
                   "val_bleu1", "val_bleu2", "val_bleu3", "val_bleu4"]:
            metric_key = k.replace("val_", "").replace("acc", "accuracy")
            history[k].append(m[metric_key])

        scheduler.step()

        logger.info(
            f"  Epoch {epoch:>2d}  loss={train_loss:.4f}/{val_loss:.4f}  "
            f"Acc={m['accuracy']:.4f}  EM={m['em']:.4f}  F1={m['f1']:.4f}  "
            f"METEOR={m['meteor']:.4f}  B4={m['bleu4']:.4f}  lr={cur_lr:.1e}"
        )

        # ── Checkpoint (best F1) ──
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            path = os.path.join(ckpt_dir, f"best_{name}.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_f1": best_f1,
                "metrics": m,
            }, path)
            logger.info(f"    ★ checkpoint saved (F1={best_f1:.4f})")

        # ── Early stopping ──
        if stopper(m["f1"]):
            logger.info(f"  Early stopping at epoch {epoch} (patience={stopper.patience})")
            break

    logger.info(f"\n{name} done. Best F1 = {best_f1:.4f}\n")
    return history
