"""Evaluation metrics for VQA: Accuracy, EM, F1, BLEU-1~4, METEOR."""

from __future__ import annotations

from collections import Counter

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score as _nltk_meteor

from src.data.preprocessing import normalize_answer, majority_answer


def compute_exact_match(pred: str, ref: str) -> float:
    """Exact match after normalization."""
    return float(normalize_answer(pred) == normalize_answer(ref))


def compute_f1(pred: str, ref: str) -> float:
    """Token-level F1 score (harmonic mean of precision and recall)."""
    pred_toks = normalize_answer(pred).split()
    ref_toks = normalize_answer(ref).split()
    if not pred_toks or not ref_toks:
        return float(pred_toks == ref_toks)
    common = Counter(pred_toks) & Counter(ref_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(ref_toks)
    return 2 * precision * recall / (precision + recall)


def compute_bleu(pred: str, ref: str) -> dict[str, float]:
    """BLEU-1 through BLEU-4 with exponential smoothing (method4)."""
    smoothie = SmoothingFunction().method4
    pred_toks = normalize_answer(pred).split()
    ref_toks = normalize_answer(ref).split()
    if not pred_toks or not ref_toks:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    return {
        "bleu1": sentence_bleu([ref_toks], pred_toks, weights=(1, 0, 0, 0), smoothing_function=smoothie),
        "bleu2": sentence_bleu([ref_toks], pred_toks, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie),
        "bleu3": sentence_bleu([ref_toks], pred_toks, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothie),
        "bleu4": sentence_bleu([ref_toks], pred_toks, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie),
    }


def compute_meteor(pred: str, ref: str) -> float:
    """METEOR score (synonym + stemming aware)."""
    pred_toks = normalize_answer(pred).split()
    ref_toks = normalize_answer(ref).split()
    if not pred_toks or not ref_toks:
        return 0.0
    return _nltk_meteor([ref_toks], pred_toks)


def compute_vqa_accuracy(pred: str, direct_answers) -> float:
    """Soft VQA accuracy: min(#annotators_agree / 3, 1.0).

    If direct_answers is a string, falls back to exact match.
    """
    if isinstance(direct_answers, str):
        return compute_exact_match(pred, direct_answers)
    normed_pred = normalize_answer(pred)
    matches = sum(1 for a in direct_answers if normalize_answer(a) == normed_pred)
    return min(matches / 3.0, 1.0)


def batch_metrics(predictions: list[str], references: list) -> dict[str, float]:
    """Compute all 8 metrics over a batch of (prediction, reference) pairs.

    Returns:
        Dict with keys: accuracy, em, f1, meteor, bleu1, bleu2, bleu3, bleu4.
    """
    ems, f1s, accs, meteors = [], [], [], []
    b1s, b2s, b3s, b4s = [], [], [], []

    for pred, ref in zip(predictions, references):
        ref_str = ref if isinstance(ref, str) else majority_answer(ref)
        ems.append(compute_exact_match(pred, ref_str))
        f1s.append(compute_f1(pred, ref_str))
        accs.append(compute_vqa_accuracy(pred, ref))
        meteors.append(compute_meteor(pred, ref_str))
        bleu = compute_bleu(pred, ref_str)
        b1s.append(bleu["bleu1"])
        b2s.append(bleu["bleu2"])
        b3s.append(bleu["bleu3"])
        b4s.append(bleu["bleu4"])

    return {
        "accuracy": float(np.mean(accs)),
        "em": float(np.mean(ems)),
        "f1": float(np.mean(f1s)),
        "meteor": float(np.mean(meteors)),
        "bleu1": float(np.mean(b1s)),
        "bleu2": float(np.mean(b2s)),
        "bleu3": float(np.mean(b3s)),
        "bleu4": float(np.mean(b4s)),
    }
