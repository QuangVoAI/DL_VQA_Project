"""Visualization utilities updated for Dual Attention (Spatial + Text)."""

from __future__ import annotations
import logging
from collections import defaultdict
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from src.data.dataset import PAD_IDX, SOS_IDX, EOS_IDX
from src.data.preprocessing import normalize_answer, majority_answer, classify_question

COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
logger = logging.getLogger("VQA")

# ═══════════════════════════════════════════════════════════════════════
# 1. Training curves
# ═══════════════════════════════════════════════════════════════════════
def plot_training_curves(all_histories: dict[str, dict[str, list[float]]], save_prefix: str = "fig") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    titles = ["Training Loss", "Validation Loss", "Learning Rate"]
    keys = ["train_loss", "val_loss", "lr"]
    
    for ax, key, title in zip(axes, keys, titles):
        for i, (name, h) in enumerate(all_histories.items()):
            ax.plot(h[key], label=name, color=COLORS[i % len(COLORS)], marker="o", ms=3)
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}1_loss_lr.png", dpi=200)
    plt.show()

# ═══════════════════════════════════════════════════════════════════════
# 2. Dual Attention Visualization
# ═══════════════════════════════════════════════════════════════════════
def visualize_attention(model, loader, answer_vocab, question_vocab, device, n=3, save_path="fig6_attn.png") -> None:
    if not model.use_attention: return
    model.eval()
    batch = next(iter(loader))
    imgs, qs, ql, _, _, ans_txt = batch
    imgs_d, qs_d, ql_d = imgs.to(device), qs.to(device), ql.to(device)
    inv_norm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])

    for idx in range(min(n, len(ans_txt))):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        img_feat = model.image_encoder(imgs_d[idx:idx+1])
        q_out, (h, c), q_mask = model.question_encoder(qs_d[idx:idx+1], ql_d[idx:idx+1])
        tok = torch.tensor([SOS_IDX], device=device)
        text_attn_list, gen_tokens = [], []

        with torch.no_grad():
            for _ in range(15):
                emb = model.answer_decoder.embedding(tok.unsqueeze(1))
                text_ctx, t_weights = model.answer_decoder.text_attention(h[-1], q_out, q_mask)
                img_ctx, s_weights = model.answer_decoder.spatial_attention(h[-1], img_feat)
                text_attn_list.append(t_weights.cpu().numpy().flatten())
                inp = torch.cat([emb, text_ctx.unsqueeze(1), img_ctx.unsqueeze(1)], 2)
                out, (h, c) = model.answer_decoder.lstm(inp, (h, c))
                pred = model.answer_decoder.fc(out.squeeze(1))
                tok = pred.argmax(1)
                gen_tokens.append(tok.item())
                if tok.item() == EOS_IDX: break

        q_toks = [question_vocab.itos.get(t, "?") for t in qs[idx].tolist() if t not in (PAD_IDX, SOS_IDX, EOS_IDX)]
        a_toks = [answer_vocab.itos.get(t, "?") for t in gen_tokens if t != EOS_IDX]
        img_show = inv_norm(imgs[idx]).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[0].imshow(img_show); axes[0].axis("off")
        axes[0].set_title(f"Q: {' '.join(q_toks)}\nPred: {' '.join(a_toks)}", fontsize=10)
        mat = np.array(text_attn_list[:len(a_toks)])[:, :len(q_toks)]
        im = axes[1].imshow(mat, cmap="YlOrRd", aspect="auto")
        axes[1].set_xticks(range(len(q_toks))); axes[1].set_xticklabels(q_toks, rotation=45)
        axes[1].set_yticks(range(len(a_toks))); axes[1].set_yticklabels(a_toks)
        plt.colorbar(im, ax=axes[1]); plt.show()

# ═══════════════════════════════════════════════════════════════════════
# 3. Missing CLI Helpers
# ═══════════════════════════════════════════════════════════════════════
def plot_radar_chart(test_results: dict[str, dict[str, float]], save_path: str = "fig4_radar.png") -> None:
    metrics = ["accuracy", "em", "f1", "meteor", "bleu4"]
    labels = ["Acc", "EM", "F1", "METEOR", "B-4"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, (name, m) in enumerate(test_results.items()):
        values = [m.get(k, 0.0) for k in metrics] + [m.get(metrics[0], 0.0)]
        ax.plot(angles, values, "o-", label=name, color=COLORS[i % len(COLORS)], linewidth=2)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels); ax.legend()
    plt.savefig(save_path); plt.show()

def plot_bar_chart(test_results: dict[str, dict[str, float]], save_path: str = "fig5_bar.png") -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(test_results))
    ax.bar(x, [m.get("f1", 0.0) for m in test_results.values()], color=COLORS[:len(test_results)])
    ax.set_xticks(x); ax.set_xticklabels(test_results.keys()); ax.set_title("F1 Comparison")
    plt.savefig(save_path); plt.show()

def plot_confusion_matrix(preds, refs, questions, save_path="fig8_cm.png"):
    # Simplified version for compatibility
    logger.info("Plotting confusion matrix placeholder...")
    plt.figure(figsize=(8,6)); plt.title("Confusion Matrix Placeholder"); plt.show()

def plot_question_type_analysis(type_results, save_path="fig9_qtype.png"):
    types = list(type_results.keys())
    f1s = [type_results[t]["f1"] for t in types]
    plt.figure(figsize=(12, 6))
    plt.bar(types, f1s, color=COLORS[1])
    plt.title("F1 Score by Question Type"); plt.xticks(rotation=45); plt.show()

def visualize_attention_overlay(*args, **kwargs):
    # Required by __init__.py
    logger.info("Spatial Overlay visualization not yet implemented for Dual Attention.")