"""Visualization utilities: training curves, radar, bar, attention overlay, confusion matrix."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from src.data.dataset import PAD_IDX, SOS_IDX, EOS_IDX
from src.data.preprocessing import normalize_answer, majority_answer, classify_question
from src.utils.metrics import compute_f1

COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]


# ═══════════════════════════════════════════════════════════════════════
# 1. Training curves
# ═══════════════════════════════════════════════════════════════════════

def plot_training_curves(
    all_histories: dict[str, dict[str, list[float]]],
    save_prefix: str = "fig",
) -> None:
    """Plot 3 figure sets: Loss+LR, Core Metrics, BLEU scores.

    Args:
        all_histories: Dict mapping model_name → history dict.
        save_prefix: Prefix for saved PNG files.
    """
    # ── Figure 1: Loss & LR ──
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    for i, (name, h) in enumerate(all_histories.items()):
        axes[0].plot(h["train_loss"], label=name, color=COLORS[i % len(COLORS)], marker="o", ms=3)
    axes[0].set_title("Training Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    for i, (name, h) in enumerate(all_histories.items()):
        axes[1].plot(h["val_loss"], label=name, color=COLORS[i % len(COLORS)], marker="o", ms=3)
    axes[1].set_title("Validation Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    for i, (name, h) in enumerate(all_histories.items()):
        axes[2].plot(h["lr"], label=name, color=COLORS[i % len(COLORS)], marker="o", ms=3)
    axes[2].set_title("Learning Rate Schedule", fontweight="bold")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}1_loss_lr.png", dpi=200, bbox_inches="tight")
    plt.show()

    # ── Figure 2: Core Metrics ──
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    metric_keys = [("val_acc", "Accuracy"), ("val_em", "Exact Match"),
                   ("val_f1", "F1"), ("val_meteor", "METEOR")]
    for ax, (key, title) in zip(axes, metric_keys):
        for i, (name, h) in enumerate(all_histories.items()):
            ax.plot(h[key], label=name, color=COLORS[i % len(COLORS)], marker="o", ms=3)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}2_core_metrics.png", dpi=200, bbox_inches="tight")
    plt.show()

    # ── Figure 3: BLEU ──
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    for ax, n in zip(axes, [1, 2, 3, 4]):
        key = f"val_bleu{n}"
        for i, (name, h) in enumerate(all_histories.items()):
            ax.plot(h[key], label=name, color=COLORS[i % len(COLORS)], marker="o", ms=3)
        ax.set_title(f"BLEU-{n}", fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel(f"BLEU-{n}")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_prefix}3_bleu.png", dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Training curve figures saved ({save_prefix}1-3).")


# ═══════════════════════════════════════════════════════════════════════
# 2. Radar & Bar Charts
# ═══════════════════════════════════════════════════════════════════════

def plot_radar_chart(
    test_results: dict[str, dict[str, float]],
    save_path: str = "fig4_radar.png",
) -> None:
    """Radar chart comparing all models across 8 metrics."""
    metrics_radar = ["accuracy", "em", "f1", "meteor", "bleu1", "bleu2", "bleu3", "bleu4"]
    labels_radar = ["Acc", "EM", "F1", "METEOR", "B-1", "B-2", "B-3", "B-4"]
    angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, (name, m) in enumerate(test_results.items()):
        values = [m[k] for k in metrics_radar] + [m[metrics_radar[0]]]
        ax.plot(angles, values, "o-", label=name, color=COLORS[i % len(COLORS)], linewidth=2)
        ax.fill(angles, values, alpha=0.08, color=COLORS[i % len(COLORS)])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_radar, fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.set_title("Test Set — All Metrics", fontsize=14, pad=20, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_bar_chart(
    test_results: dict[str, dict[str, float]],
    save_path: str = "fig5_bar.png",
) -> None:
    """Grouped bar chart comparing metrics across models."""
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(test_results))
    w = 0.1
    bar_metrics = ["accuracy", "em", "f1", "meteor", "bleu1", "bleu2", "bleu3", "bleu4"]
    bar_labels = ["Acc", "EM", "F1", "METEOR", "B-1", "B-2", "B-3", "B-4"]
    bar_colors = ["#e74c3c", "#f39c12", "#2ecc71", "#8e44ad",
                  "#3498db", "#9b59b6", "#1abc9c", "#e67e22"]
    for j, (mk, ml, mc) in enumerate(zip(bar_metrics, bar_labels, bar_colors)):
        vals = [test_results[n][mk] for n in test_results]
        ax.bar(x + (j - 3.5) * w, vals, w, label=ml, color=mc, alpha=0.85)
    ax.set_title("Test Set — All Metrics Comparison", fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in test_results], fontsize=9)
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ═══════════════════════════════════════════════════════════════════════
# 3. Attention Visualization — Heatmap over question tokens
# ═══════════════════════════════════════════════════════════════════════

def visualize_attention(
    model: torch.nn.Module,
    loader: Any,
    answer_vocab: Any,
    question_vocab: Any,
    device: torch.device,
    n: int = 3,
    save_path: str = "fig6_attention_heatmap.png",
) -> None:
    """Visualize attention weights as a heatmap (answer tokens × question tokens).

    Args:
        model: VQAModel with use_attention=True.
        loader: DataLoader.
        answer_vocab: Answer vocabulary.
        question_vocab: Question vocabulary.
        device: Torch device.
        n: Number of examples.
        save_path: Output file path.
    """
    if not model.use_attention:
        print("This model does not use attention.")
        return

    model.eval()
    batch = next(iter(loader))
    imgs, qs, ql, ans, al, ans_txt = batch
    imgs_d, qs_d, ql_d = imgs.to(device), qs.to(device), ql.to(device)

    inv_norm = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for idx in range(min(n, len(ans_txt))):
        img_feat = model.image_encoder(imgs_d[idx : idx + 1])
        q_out, (h, c), q_mask = model.question_encoder(qs_d[idx : idx + 1], ql_d[idx : idx + 1])

        tok = torch.tensor([SOS_IDX], device=device)
        attn_matrix, gen_tokens = [], []

        with torch.no_grad():
            for _ in range(15):
                emb = model.answer_decoder.embedding(tok.unsqueeze(1))
                ctx, attn_w = model.answer_decoder.attention(h[-1], q_out, q_mask)
                attn_matrix.append(attn_w.cpu().numpy().flatten())
                inp = torch.cat([emb, ctx.unsqueeze(1), img_feat.unsqueeze(1)], 2)
                out, (h, c) = model.answer_decoder.lstm(inp, (h, c))
                pred = model.answer_decoder.fc(out.squeeze(1))
                tok = pred.argmax(1)
                gen_tokens.append(tok.item())
                if tok.item() == EOS_IDX:
                    break

        q_toks = [
            question_vocab.itos.get(t, "?")
            for t in qs[idx].tolist()
            if t not in (PAD_IDX, SOS_IDX) and t != question_vocab.stoi["<EOS>"]
        ]
        a_toks = [
            answer_vocab.itos.get(t, "?")
            for t in gen_tokens
            if t not in (PAD_IDX, SOS_IDX, EOS_IDX)
        ]

        # Image
        img_show = inv_norm(imgs[idx]).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[idx, 0].imshow(img_show)
        axes[idx, 0].set_title(
            f"Q: {' '.join(q_toks)}\nGT: {ans_txt[idx]}\nPred: {' '.join(a_toks)}", fontsize=9
        )
        axes[idx, 0].axis("off")

        # Attention heatmap
        mat = np.array(attn_matrix[: len(a_toks)])
        if mat.ndim == 2 and mat.shape[1] >= len(q_toks):
            mat = mat[:, : len(q_toks)]
        im = axes[idx, 1].imshow(mat, cmap="YlOrRd", aspect="auto")
        axes[idx, 1].set_xticks(range(len(q_toks)))
        axes[idx, 1].set_xticklabels(q_toks, rotation=45, ha="right", fontsize=8)
        axes[idx, 1].set_yticks(range(len(a_toks)))
        axes[idx, 1].set_yticklabels(a_toks, fontsize=8)
        axes[idx, 1].set_xlabel("Question tokens")
        axes[idx, 1].set_ylabel("Generated tokens")
        axes[idx, 1].set_title("Attention Weights", fontsize=10)
        plt.colorbar(im, ax=axes[idx, 1], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# 4. Attention Overlay on Image (spatial-style visualization)
# ═══════════════════════════════════════════════════════════════════════

def visualize_attention_overlay(
    model: torch.nn.Module,
    loader: Any,
    answer_vocab: Any,
    question_vocab: Any,
    device: torch.device,
    n: int = 3,
    save_path: str = "fig7_attention_overlay.png",
) -> None:
    """Overlay aggregated attention weights on the input image.

    Since Bahdanau attention is over *question tokens* (not spatial), we:
    1. Compute attention weights for each generated answer token.
    2. Aggregate by summing attention over all decode steps → highlight which
       question words were most attended to overall.
    3. Display as a bar-over-image layout: image + attention bar for question tokens.

    For spatial attention models, this would overlay directly on the image.
    Here we show the image alongside a question-token importance bar.

    Args:
        model: VQAModel with attention.
        loader: DataLoader.
        answer_vocab: Answer vocabulary.
        question_vocab: Question vocabulary.
        device: Torch device.
        n: Number of examples.
        save_path: Output file path.
    """
    if not model.use_attention:
        print("This model does not use attention.")
        return

    model.eval()
    batch = next(iter(loader))
    imgs, qs, ql, ans, al, ans_txt = batch
    imgs_d, qs_d, ql_d = imgs.to(device), qs.to(device), ql.to(device)

    inv_norm = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    fig, axes = plt.subplots(n, 3, figsize=(18, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for idx in range(min(n, len(ans_txt))):
        img_feat = model.image_encoder(imgs_d[idx : idx + 1])
        q_out, (h, c), q_mask = model.question_encoder(qs_d[idx : idx + 1], ql_d[idx : idx + 1])

        tok = torch.tensor([SOS_IDX], device=device)
        attn_weights_all, gen_tokens = [], []

        with torch.no_grad():
            for _ in range(15):
                emb = model.answer_decoder.embedding(tok.unsqueeze(1))
                ctx, attn_w = model.answer_decoder.attention(h[-1], q_out, q_mask)
                attn_weights_all.append(attn_w.cpu().numpy().flatten())
                inp = torch.cat([emb, ctx.unsqueeze(1), img_feat.unsqueeze(1)], 2)
                out, (h, c) = model.answer_decoder.lstm(inp, (h, c))
                pred = model.answer_decoder.fc(out.squeeze(1))
                tok = pred.argmax(1)
                gen_tokens.append(tok.item())
                if tok.item() == EOS_IDX:
                    break

        q_toks = [
            question_vocab.itos.get(t, "?")
            for t in qs[idx].tolist()
            if t not in (PAD_IDX, SOS_IDX) and t != question_vocab.stoi["<EOS>"]
        ]
        a_toks = [
            answer_vocab.itos.get(t, "?")
            for t in gen_tokens
            if t not in (PAD_IDX, SOS_IDX, EOS_IDX)
        ]

        # Aggregated attention (sum over decode steps, then normalize)
        attn_agg = np.sum(attn_weights_all, axis=0)
        if len(attn_agg) >= len(q_toks):
            attn_agg = attn_agg[: len(q_toks)]
        attn_agg = attn_agg / (attn_agg.sum() + 1e-9)

        # Panel 1: Image
        img_show = inv_norm(imgs[idx]).clamp(0, 1).permute(1, 2, 0).numpy()
        axes[idx, 0].imshow(img_show)
        axes[idx, 0].set_title(f"Q: {' '.join(q_toks)}", fontsize=9)
        axes[idx, 0].axis("off")

        # Panel 2: Question token importance (horizontal bar)
        y_pos = np.arange(len(q_toks))
        bar_colors = plt.cm.YlOrRd(attn_agg / (attn_agg.max() + 1e-9))
        axes[idx, 1].barh(y_pos, attn_agg, color=bar_colors, edgecolor="gray", linewidth=0.5)
        axes[idx, 1].set_yticks(y_pos)
        axes[idx, 1].set_yticklabels(q_toks, fontsize=9)
        axes[idx, 1].invert_yaxis()
        axes[idx, 1].set_xlabel("Aggregated Attention Weight")
        axes[idx, 1].set_title("Question Token Importance", fontsize=10, fontweight="bold")
        axes[idx, 1].grid(alpha=0.3, axis="x")

        # Panel 3: Image with text overlay showing attention-highlighted question
        axes[idx, 2].imshow(img_show, alpha=0.7)
        axes[idx, 2].axis("off")

        # Highlight important question words by size/color
        text_parts = []
        for w, a_val in zip(q_toks, attn_agg):
            text_parts.append((w, a_val))

        y_text = 0.95
        title_text = "GT: " + ans_txt[idx][:60] + "\nPred: " + " ".join(a_toks)[:60]
        axes[idx, 2].set_title(title_text, fontsize=9)

        # Word cloud–style layout on image
        x_pos_text = 0.05
        for w, a_val in text_parts:
            font_size = max(8, min(20, int(8 + a_val * 60)))
            alpha_val = max(0.4, min(1.0, 0.3 + a_val * 3))
            axes[idx, 2].text(
                x_pos_text, y_text, w, transform=axes[idx, 2].transAxes,
                fontsize=font_size, fontweight="bold",
                color=plt.cm.Reds(0.3 + a_val * 0.7),
                alpha=alpha_val,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.6),
            )
            x_pos_text += max(0.06, len(w) * 0.018 + 0.02)
            if x_pos_text > 0.90:
                x_pos_text = 0.05
                y_text -= 0.08

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# 5. Confusion Matrix (predicted vs reference answer type)
# ═══════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    preds: list[str],
    refs: list[str],
    questions: list[str],
    save_path: str = "fig8_confusion_matrix.png",
) -> None:
    """Plot a confusion matrix of question types vs prediction correctness.

    Rows = question types, columns = [Correct, Incorrect].
    Also shows per-type F1 as a stacked bar chart.

    Args:
        preds: Predicted answers.
        refs: Reference answers.
        questions: Question strings.
        save_path: Output file path.
    """
    type_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "incorrect": 0, "total": 0})

    for p, r, q in zip(preds, refs, questions):
        qtype = classify_question(q)
        ref_str = r if isinstance(r, str) else majority_answer(r)
        is_correct = normalize_answer(p) == normalize_answer(ref_str)
        type_stats[qtype]["total"] += 1
        if is_correct:
            type_stats[qtype]["correct"] += 1
        else:
            type_stats[qtype]["incorrect"] += 1

    # Sort by total count
    sorted_types = sorted(type_stats.items(), key=lambda x: -x[1]["total"])
    labels = [t for t, _ in sorted_types]
    correct = [s["correct"] for _, s in sorted_types]
    incorrect = [s["incorrect"] for _, s in sorted_types]
    totals = [s["total"] for _, s in sorted_types]
    accuracies = [c / t if t > 0 else 0 for c, t in zip(correct, totals)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Stacked bar chart
    y_pos = np.arange(len(labels))
    ax1.barh(y_pos, correct, color="#2ecc71", label="Correct", edgecolor="white")
    ax1.barh(y_pos, incorrect, left=correct, color="#e74c3c", label="Incorrect", edgecolor="white")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{l} (n={t})" for l, t in zip(labels, totals)], fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("Number of Questions")
    ax1.set_title("Correct vs Incorrect by Question Type", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3, axis="x")

    # Accuracy bar
    bar_colors = [plt.cm.RdYlGn(a) for a in accuracies]
    ax2.barh(y_pos, accuracies, color=bar_colors, edgecolor="gray", linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel("Exact Match Accuracy")
    ax2.set_title("Accuracy by Question Type", fontweight="bold")
    ax2.set_xlim(0, 1.0)
    ax2.grid(alpha=0.3, axis="x")

    # Add percentage labels
    for i, (acc, total) in enumerate(zip(accuracies, totals)):
        ax2.text(acc + 0.01, i, f"{acc:.1%}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# 6. Question Type Analysis (detailed metrics per type)
# ═══════════════════════════════════════════════════════════════════════

def plot_question_type_analysis(
    type_results: dict[str, dict[str, float]],
    save_path: str = "fig9_question_type_analysis.png",
) -> None:
    """Grouped bar chart showing EM, F1, METEOR per question type.

    Args:
        type_results: Output from ``evaluate_by_question_type()``.
        save_path: Output file path.
    """
    types = list(type_results.keys())
    ems = [type_results[t]["em"] for t in types]
    f1s = [type_results[t]["f1"] for t in types]
    meteors = [type_results[t]["meteor"] for t in types]
    totals = [type_results[t]["total"] for t in types]

    x = np.arange(len(types))
    w = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w, ems, w, label="Exact Match", color="#e74c3c", alpha=0.85)
    ax.bar(x, f1s, w, label="F1", color="#3498db", alpha=0.85)
    ax.bar(x + w, meteors, w, label="METEOR", color="#2ecc71", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={n})" for t, n in zip(types, totals)], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Metrics by Question Type", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")
