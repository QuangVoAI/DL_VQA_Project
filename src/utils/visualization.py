"""Visualization utilities for VQA (Local Run — Windows compatible)."""

from __future__ import annotations
import logging
import os
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

_INV_NORM = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


def plot_training_curves(
    all_histories: dict[str, dict[str, list[float]]],
    save_prefix: str = "fig",
) -> None:
    """Vẽ 3 đồ thị: Train Loss, Val Loss, Learning Rate theo epoch."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    titles = ["Training Loss", "Validation Loss", "Learning Rate"]
    keys   = ["train_loss",   "val_loss",        "lr"]

    for ax, key, title in zip(axes, keys, titles):
        for i, (name, h) in enumerate(all_histories.items()):
            vals = h.get(key, [])
            if vals:
                ax.plot(vals, label=name, color=COLORS[i % len(COLORS)], marker="o", ms=3)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = f"{save_prefix}1_loss_lr.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

    # Val F1 theo epoch
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for i, (name, h) in enumerate(all_histories.items()):
        vals = h.get("val_f1", [])
        if vals:
            ax2.plot(vals, label=name, color=COLORS[i % len(COLORS)], marker="o", ms=3)
    ax2.set_title("Validation F1 Score", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    save_path2 = f"{save_prefix}2_val_f1.png"
    plt.savefig(save_path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {save_path2}")
def visualize_attention(
    model, loader, answer_vocab, question_vocab, device,
    n: int = 3, save_path: str = "fig6_attn.png",
) -> None:
    if not model.use_attention:
        print("Model không có attention, bỏ qua visualize_attention.")
        return

    model.eval()
    batch = next(iter(loader))
    imgs, qs, ql, _, _, ans_txt = batch
    imgs_d = imgs.to(device)
    qs_d   = qs.to(device)
    ql_d   = ql.to(device)

    collected_figs: list[plt.Figure] = []

    for idx in range(min(n, imgs.size(0))):
        with torch.no_grad():
            img_feat = model.image_encoder(imgs_d[idx : idx + 1])            # (1, 49, D)
            q_out, (h, c), _ = model.question_encoder(
                qs_d[idx : idx + 1], ql_d[idx : idx + 1]
            )  # q_out: (1, q_len_packed, H)

        # Recompute q_mask to match q_out's actual sequence length (max(lengths) for this sub-batch)
        actual_q_len = q_out.size(1)
        sample_len   = int(ql_d[idx].item())
        q_mask = torch.zeros(1, actual_q_len, device=device)
        q_mask[0, : min(sample_len, actual_q_len)] = 1.0

        # Collect readable question tokens (no special tokens)
        q_ids = qs[idx].tolist()
        q_toks = [
            question_vocab.itos.get(t, "?")
            for t in q_ids
            if t not in (PAD_IDX, SOS_IDX, EOS_IDX)
        ]
        q_len = actual_q_len  # padded sequence length from encoder

        text_attn_list: list[np.ndarray] = []
        gen_tokens: list[int] = []

        tok = torch.tensor([SOS_IDX], device=device)
        for _ in range(30):
            with torch.no_grad():
                emb = model.answer_decoder.embedding(tok.unsqueeze(1))          # (1,1,E)
                text_ctx, t_weights = model.answer_decoder.text_attention(
                    h[-1], q_out, q_mask
                )  # t_weights: (1, q_len)
                img_ctx, _ = model.answer_decoder.spatial_attention(h[-1], img_feat)

                # Lưu attention weights — luôn là (q_len,) sau khi squeeze
                tw = t_weights.squeeze(0).cpu().numpy()  # (q_len,)
                text_attn_list.append(tw)

                inp = torch.cat([emb, text_ctx.unsqueeze(1), img_ctx.unsqueeze(1)], dim=2)
                out, (h, c) = model.answer_decoder.lstm(inp, (h, c))
                residual    = model.answer_decoder.res_proj(inp.squeeze(1))
                out_res     = model.answer_decoder.layer_norm(out.squeeze(1) + residual)
                pred        = model.answer_decoder.fc(model.answer_decoder.fc_drop(out_res))
                tok         = pred.argmax(1)

            gen_tokens.append(tok.item())
            if tok.item() == EOS_IDX:
                break

        a_toks = [
            answer_vocab.itos.get(t, "?")
            for t in gen_tokens
            if t not in (EOS_IDX, PAD_IDX)
        ]

        if not q_toks or not a_toks:
            continue

        # Build attention matrix: (n_ans_toks, q_len)
        mat = np.array(text_attn_list[: len(a_toks)])  # (A, q_len)
        # Trim columns to actual (non-padded) question tokens
        if mat.shape[1] > len(q_toks):
            mat = mat[:, : len(q_toks)]
        effective_q = mat.shape[1]

        img_show = _INV_NORM(imgs[idx]).clamp(0, 1).permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].imshow(img_show)
        axes[0].axis("off")
        axes[0].set_title(
            f"Q: {' '.join(q_toks)}\nPred: {' '.join(a_toks)}", fontsize=9
        )

        im = axes[1].imshow(mat, cmap="YlOrRd", aspect="auto")
        axes[1].set_xticks(range(effective_q))
        axes[1].set_xticklabels(q_toks[:effective_q], rotation=45, ha="right", fontsize=8)
        axes[1].set_yticks(range(len(a_toks)))
        axes[1].set_yticklabels(a_toks, fontsize=8)
        axes[1].set_title("Text Attention Weights")
        plt.colorbar(im, ax=axes[1])
        plt.tight_layout()
        collected_figs.append(fig)

    if collected_figs:
        collected_figs[0].savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        for f in collected_figs:
            plt.figure(f.number)
            plt.show()
            plt.close(f)
    else:
        print("visualize_attention: không có sample nào hợp lệ để vẽ.")
def visualize_attention_overlay(
    model, loader, answer_vocab, question_vocab, device,
    n: int = 3, save_path: str = "fig7_overlay.png",
) -> None:
    if not model.use_attention:
        print("Model không có attention, bỏ qua visualize_attention_overlay.")
        return

    model.eval()
    batch = next(iter(loader))
    imgs, qs, ql, _, _, ans_txt = batch
    imgs_d = imgs.to(device)
    qs_d   = qs.to(device)
    ql_d   = ql.to(device)

    collected_figs: list[plt.Figure] = []

    for idx in range(min(n, imgs.size(0))):
        with torch.no_grad():
            img_feat = model.image_encoder(imgs_d[idx : idx + 1])  # (1, 49, D)
            q_out, (h, c), _ = model.question_encoder(
                qs_d[idx : idx + 1], ql_d[idx : idx + 1]
            )

        # Recompute q_mask to match q_out's actual packed sequence length
        actual_q_len = q_out.size(1)
        sample_len   = int(ql_d[idx].item())
        q_mask = torch.zeros(1, actual_q_len, device=device)
        q_mask[0, : min(sample_len, actual_q_len)] = 1.0

        spatial_attn_maps: list[np.ndarray] = []
        gen_tokens: list[int] = []

        tok = torch.tensor([SOS_IDX], device=device)
        for _ in range(30):
            with torch.no_grad():
                emb = model.answer_decoder.embedding(tok.unsqueeze(1))
                text_ctx, _ = model.answer_decoder.text_attention(h[-1], q_out, q_mask)
                img_ctx, s_weights = model.answer_decoder.spatial_attention(h[-1], img_feat)
                # s_weights: (1, 49) → squeeze → (49,)
                spatial_attn_maps.append(s_weights.squeeze().cpu().numpy())

                inp = torch.cat([emb, text_ctx.unsqueeze(1), img_ctx.unsqueeze(1)], dim=2)
                out, (h, c) = model.answer_decoder.lstm(inp, (h, c))
                residual    = model.answer_decoder.res_proj(inp.squeeze(1))
                out_res     = model.answer_decoder.layer_norm(out.squeeze(1) + residual)
                pred        = model.answer_decoder.fc(model.answer_decoder.fc_drop(out_res))
                tok         = pred.argmax(1)

            gen_tokens.append(tok.item())
            if tok.item() == EOS_IDX:
                break

        a_toks = [answer_vocab.itos.get(t, "?") for t in gen_tokens if t not in (EOS_IDX, PAD_IDX)]
        q_toks = [
            question_vocab.itos.get(t, "?")
            for t in qs[idx].tolist()
            if t not in (PAD_IDX, SOS_IDX, EOS_IDX)
        ]

        img_show = _INV_NORM(imgs[idx]).clamp(0, 1).permute(1, 2, 0).numpy()

        if not spatial_attn_maps:
            continue

        # Average spatial attention across decode steps
        raw_maps = np.stack(spatial_attn_maps, axis=0)   # (steps, 49)
        avg_attn  = raw_maps.mean(axis=0)                 # (49,)

        n_regions = avg_attn.shape[0]
        grid_size = int(round(np.sqrt(n_regions)))  # typically 7
        if grid_size * grid_size != n_regions:
            # fallback: pick nearest square
            grid_size = int(np.sqrt(n_regions))
            avg_attn = avg_attn[: grid_size * grid_size]

        attn_map = avg_attn.reshape(grid_size, grid_size)

        H, W = img_show.shape[:2]
        # Upsample attention map to image size using kron
        scale_h = (H + grid_size - 1) // grid_size
        scale_w = (W + grid_size - 1) // grid_size
        attn_full = np.kron(attn_map / (attn_map.max() + 1e-8), np.ones((scale_h, scale_w)))
        attn_full = attn_full[:H, :W]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].imshow(img_show)
        axes[0].axis("off")
        axes[0].set_title(f"Q: {' '.join(q_toks)}\nA: {' '.join(a_toks)}", fontsize=9)

        axes[1].imshow(attn_map, cmap="hot", interpolation="bilinear")
        axes[1].set_title("Spatial Attention Heatmap")
        axes[1].axis("off")

        axes[2].imshow(img_show)
        axes[2].imshow(attn_full, cmap="jet", alpha=0.45)
        axes[2].set_title("Spatial Attention Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        collected_figs.append(fig)

    if collected_figs:
        collected_figs[0].savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
        for f in collected_figs:
            plt.figure(f.number)
            plt.show()
            plt.close(f)
    else:
        print("visualize_attention_overlay: không có sample nào hợp lệ để vẽ.")
def plot_radar_chart(
    test_results: dict[str, dict[str, float]],
    save_path: str = "fig4_radar.png",
) -> None:
    """Radar chart so sánh các metric chính giữa các mô hình."""
    # Dùng 4 độ đo chính cho rationale: F1, METEOR, ROUGE-L, BLEU-4
    metrics = ["f1", "meteor", "rouge_l", "bleu4"]
    labels  = ["F1", "METEOR", "R-L", "B-4"]
    angles  = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, (name, m) in enumerate(test_results.items()):
        values = [m.get(k, 0.0) for k in metrics] + [m.get(metrics[0], 0.0)]
        ax.plot(angles, values, "o-", label=name, color=COLORS[i % len(COLORS)], linewidth=2)
        ax.fill(angles, values, alpha=0.1, color=COLORS[i % len(COLORS)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)
    ax.set_title("Model Comparison (Radar)", size=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {save_path}")


def plot_bar_chart(
    test_results: dict[str, dict[str, float]],
    save_path: str = "fig5_bar.png",
) -> None:
    """Bar chart so sánh F1, METEOR, ROUGE-L, BLEU-4 giữa các mô hình."""
    metric_groups = {
        "F1":       "f1",
        "METEOR":   "meteor",
        "ROUGE-L":  "rouge_l",
        "BLEU-4":   "bleu4",
    }
    names     = list(test_results.keys())
    x         = np.arange(len(names))
    bar_width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    for j, (label, key) in enumerate(metric_groups.items()):
        vals = [test_results[n].get(key, 0.0) for n in names]
        ax.bar(x + j * bar_width, vals, bar_width, label=label,
               color=COLORS[j % len(COLORS)], alpha=0.85)

    ax.set_xticks(x + bar_width * (len(metric_groups) - 1) / 2)
    ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {save_path}")
def plot_question_type_analysis(
    type_results: dict, save_path: str = "fig9_qtype.png"
) -> None:
    if not type_results:
        print("Không có dữ liệu question type analysis.")
        return

    types  = list(type_results.keys())
    f1s    = [type_results[t]["f1"] for t in types]
    ems    = [type_results[t]["em"] for t in types]
    counts = [type_results[t]["total"] for t in types]
    x      = np.arange(len(types))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    bars1 = axes[0].bar(x, f1s, color=COLORS[1], alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(types, rotation=30, ha="right")
    axes[0].set_title("F1 by Question Type", fontweight="bold")
    axes[0].set_ylabel("F1 Score")
    axes[0].grid(axis="y", alpha=0.3)
    for bar, cnt in zip(bars1, counts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"n={cnt}", ha="center", fontsize=8,
        )

    axes[1].bar(x, ems, color=COLORS[0], alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(types, rotation=30, ha="right")
    axes[1].set_title("EM by Question Type", fontweight="bold")
    axes[1].set_ylabel("Exact Match")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {save_path}")
def plot_confusion_matrix(
    preds: list, refs: list, questions: list,
    save_path: str = "fig8_cm.png",
) -> None:
    """Stacked bar: tỷ lệ đúng/sai theo question type."""
    type_correct: dict[str, list[float]] = defaultdict(list)
    for p, r, q in zip(preds, refs, questions):
        qtype   = classify_question(q)
        ref_str = r if isinstance(r, str) else majority_answer(r)
        is_correct = float(normalize_answer(p) == normalize_answer(ref_str))
        type_correct[qtype].append(is_correct)

    types          = sorted(type_correct.keys())
    correct_rates  = [float(np.mean(type_correct[t])) for t in types]
    error_rates    = [1.0 - r for r in correct_rates]
    x              = np.arange(len(types))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, correct_rates, color="#2ecc71", alpha=0.85, label="Correct (EM)")
    ax.bar(x, error_rates, bottom=correct_rates, color="#e74c3c", alpha=0.85, label="Wrong")
    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=30, ha="right")
    ax.set_title("Correct vs Wrong by Question Type", fontweight="bold")
    ax.set_ylabel("Proportion")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"Saved: {save_path}")