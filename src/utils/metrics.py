"""Evaluation metrics for VQA: Accuracy, EM, F1, BLEU-1~4, METEOR."""

from __future__ import annotations
from collections import Counter
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score as _nltk_meteor

from src.data.preprocessing import normalize_answer, majority_answer

def compute_exact_match(pred: str, ref: str) -> float:
    """So khớp chính xác sau khi đã chuẩn hóa văn bản."""
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

def compute_bleu(pred: str, ref: str) -> dict[str, float]:
    """Tính BLEU từ 1 đến 4 với làm mượt (Smoothing Method 4)."""
    smoothie = SmoothingFunction().method4
    p_toks = normalize_answer(pred).split()
    r_toks = normalize_answer(ref).split()
    if not p_toks or not r_toks:
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
    
    weights = [
        (1, 0, 0, 0),          # BLEU-1
        (0.5, 0.5, 0, 0),      # BLEU-2
        (0.33, 0.33, 0.33, 0), # BLEU-3
        (0.25, 0.25, 0.25, 0.25) # BLEU-4
    ]
    
    return {
        f"bleu{i+1}": sentence_bleu([r_toks], p_toks, weights=w, smoothing_function=smoothie)
        for i, w in enumerate(weights)
    }

def compute_meteor(pred: str, ref: str) -> float:
    """Tính METEOR score (hỗ trợ từ đồng nghĩa và biến thể từ)."""
    p_toks = normalize_answer(pred).split()
    r_toks = normalize_answer(ref).split()
    if not p_toks or not r_toks:
        return 0.0
    return _nltk_meteor([r_toks], p_toks)

def compute_vqa_accuracy(pred: str, direct_answers) -> float:
    """
    Tính VQA Accuracy mềm: min(#người_cùng_đáp_án / 3, 1.0).
    Sử dụng cho các tập dữ liệu có nhiều người gắn nhãn (như A-OKVQA).
    """
    if isinstance(direct_answers, str):
        return compute_exact_match(pred, direct_answers)
    
    normed_pred = normalize_answer(pred)
    matches = sum(1 for a in direct_answers if normalize_answer(a) == normed_pred)
    return min(matches / 3.0, 1.0)

def batch_metrics(predictions: list[str], references: list) -> dict[str, float]:
    """Tổng hợp 8 chỉ số đo lường trên toàn bộ batch dữ liệu."""
    results = {
        "accuracy": [], "em": [], "f1": [], "meteor": [],
        "bleu1": [], "bleu2": [], "bleu3": [], "bleu4": []
    }

    for pred, ref in zip(predictions, references):
        # Lấy đáp án phổ biến nhất làm mốc so sánh (nếu ref là list)
        ref_str = ref if isinstance(ref, str) else majority_answer(ref)
        
        results["accuracy"].append(compute_vqa_accuracy(pred, ref))
        results["em"].append(compute_exact_match(pred, ref_str))
        results["f1"].append(compute_f1(pred, ref_str))
        results["meteor"].append(compute_meteor(pred, ref_str))
        
        bleus = compute_bleu(pred, ref_str)
        for k, v in bleus.items():
            results[k].append(v)

    # Trả về giá trị trung bình của toàn batch
    return {k: float(np.mean(v)) for k, v in results.items()}