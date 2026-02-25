"""Standard metrics for VQA evaluation."""
from collections import Counter
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score as _nltk_meteor
from src.data.preprocessing import normalize_answer, majority_answer

def compute_f1(pred: str, ref: str) -> float:
    p_toks, r_toks = normalize_answer(pred).split(), normalize_answer(ref).split()
    if not p_toks or not r_toks: return float(p_toks == r_toks)
    common = Counter(p_toks) & Counter(r_toks)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision, recall = num_same / len(p_toks), num_same / len(r_toks)
    return 2 * precision * recall / (precision + recall)

def batch_metrics(predictions: list[str], references: list) -> dict[str, float]:
    f1s, accs, b4s = [], [], []
    for p, r in zip(predictions, references):
        ref_str = r if isinstance(r, str) else majority_answer(r)
        f1s.append(compute_f1(p, ref_str))
        # ... tính các chỉ số khác tương tự
    return {"f1": float(np.mean(f1s)), "accuracy": 0.0, "bleu4": 0.0} # v.v.