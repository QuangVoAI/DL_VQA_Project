"""Evaluation metrics for VQA: F1, BLEU-4, METEOR, ROUGE-L."""

from __future__ import annotations
from collections import Counter
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score as _nltk_meteor

from src.data.preprocessing import normalize_answer, majority_answer


def compute_exact_match(pred: str, ref: str) -> float:
    """So khớp chính xác sau khi đã chuẩn hóa văn bản (dùng cho EM / Acc nếu cần)."""
    return float(normalize_answer(pred) == normalize_answer(ref))

def compute_f1(pred: str, ref: str) -> float:
    """Tính F1-score ở mức độ token (word-level)."""
    p_toks = normalize_answer(pred).split()
    r_toks = normalize_answer(ref).split()
    if not p_toks or not r_toks:
        return float(p_toks == r_toks)
    
    common = Counter(p_toks) & Counter(r_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(p_toks)
    recall = num_same / len(r_toks)
    return 2 * precision * recall / (precision + recall)

def compute_bleu4(pred: str, ref: str) -> float:
    """Tính BLEU-4 với làm mượt (Smoothing Method 4)."""
    smoothie = SmoothingFunction().method4
    p_toks = normalize_answer(pred).split()
    r_toks = normalize_answer(ref).split()
    if not p_toks or not r_toks:
        return 0.0

    weights = (0.25, 0.25, 0.25, 0.25)
    return float(sentence_bleu([r_toks], p_toks, weights=weights, smoothing_function=smoothie))

def compute_meteor(pred: str, ref: str) -> float:
    """Tính METEOR score (hỗ trợ từ đồng nghĩa và biến thể từ)."""
    p_toks = normalize_answer(pred).split()
    r_toks = normalize_answer(ref).split()
    if not p_toks or not r_toks:
        return 0.0
    return float(_nltk_meteor([r_toks], p_toks))


def compute_vqa_accuracy(pred: str, direct_answers) -> float:
    """
    Tính VQA Accuracy mềm: min(#người_cùng_đáp_án / 3, 1.0).
    Giữ lại cho mục đích phân tích, dù batch_metrics không còn dùng.
    """
    if isinstance(direct_answers, str):
        return compute_exact_match(pred, direct_answers)

    normed_pred = normalize_answer(pred)
    matches = sum(1 for a in direct_answers if normalize_answer(a) == normed_pred)
    return min(matches / 3.0, 1.0)


def compute_bleu(pred: str, ref: str) -> dict[str, float]:
    """
    Hàm BLEU tổng hợp cho tương thích ngược.
    Trả về dict với BLEU-4 là giá trị chính, BLEU-1~3 đặt 0.0 (không còn dùng).
    """
    b4 = compute_bleu4(pred, ref)
    return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": b4}


def _lcs_length(x: list[str], y: list[str]) -> int:
    """Độ dài Longest Common Subsequence giữa hai dãy token."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(pred: str, ref: str) -> float:
    """
    ROUGE-L dựa trên LCS F1-score (phiên bản đơn giản).
    Phù hợp để đo mức độ trùng khớp cho câu dài (rationales).
    """
    p_toks = normalize_answer(pred).split()
    r_toks = normalize_answer(ref).split()
    if not p_toks or not r_toks:
        return 0.0

    lcs = _lcs_length(p_toks, r_toks)
    prec = lcs / len(p_toks)
    rec = lcs / len(r_toks)
    if prec == 0.0 or rec == 0.0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def batch_metrics(predictions: list[str], references: list) -> dict[str, float]:
    """
    Tổng hợp các chỉ số:
      - F1
      - METEOR
      - ROUGE-L
      - BLEU-4
    """
    results = {"f1": [], "meteor": [], "rouge_l": [], "bleu4": []}

    for pred, ref in zip(predictions, references):
        # Lấy đáp án chuẩn (nếu ref là list thì dùng majority_answer)
        ref_str = ref if isinstance(ref, str) else majority_answer(ref)

        results["f1"].append(compute_f1(pred, ref_str))
        results["meteor"].append(compute_meteor(pred, ref_str))
        results["rouge_l"].append(compute_rouge_l(pred, ref_str))
        results["bleu4"].append(compute_bleu4(pred, ref_str))

    # Trả về giá trị trung bình của toàn batch
    return {k: float(np.mean(v)) for k, v in results.items()}