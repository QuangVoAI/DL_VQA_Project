"""Text preprocessing utilities for VQA."""

from __future__ import annotations
import re
import string
from collections import Counter
from typing import Any

# -- Contraction mapping --
CONTRACTIONS: dict[str, str] = {
    "aint": "ain't", "arent": "aren't", "cant": "can't",
    "couldve": "could've", "couldnt": "couldn't",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't",
    "isnt": "isn't", "mightve": "might've", "mustve": "must've",
    "neednt": "needn't", "shouldve": "should've", "shouldnt": "shouldn't",
    "wasnt": "wasn't", "werent": "weren't", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't",
}
ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
PUNCTUATION = re.compile(f"[{re.escape(string.punctuation)}]")

def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation / articles / extra whitespace."""
    s = str(s).lower().strip()
    for k, v in CONTRACTIONS.items():
        s = s.replace(k, v)
    s = PUNCTUATION.sub(" ", s)
    s = ARTICLES.sub(" ", s)
    return " ".join(s.split())

def majority_answer(direct_answers: list[str]) -> str:
    """Pick the most common annotator answer."""
    if not direct_answers: return ""
    normed = [normalize_answer(a) for a in direct_answers]
    return Counter(normed).most_common(1)[0][0]

def best_rationale(rationales: list[str]) -> str:
    """Pick the longest rationale as the generation target."""
    if not rationales: return ""
    return max(rationales, key=lambda r: len(r.split()))

def extract_answer(item: dict[str, Any]) -> str:
    """Priority: _target_answer -> rationale -> direct_answers."""
    if "_target_answer" in item:
        return item["_target_answer"]
    rationales = item.get("rationales", [])
    if rationales:
        return best_rationale(rationales)
    direct_answers = item.get("direct_answers", [])
    if direct_answers:
        return majority_answer(direct_answers)
    return ""

def expand_data_with_rationales(data_list: list[dict]) -> list[dict]:
    """Expand training data by using ALL 3 rationales (~3x)."""
    expanded: list[dict] = []
    for item in data_list:
        rationales = item.get("rationales", [])
        if rationales:
            for rat in rationales:
                new_item = dict(item)
                new_item["_target_answer"] = rat
                expanded.append(new_item)
        else:
            expanded.append(item)
    return expanded

def classify_question(question: str) -> str:
    """Categorize questions for detailed analysis."""
    q = question.lower().strip()
    words = q.split()
    if not words: return "other"
    first = words[0]
    if any(kw in q for kw in ["how many", "how much", "count"]): return "counting"
    if first in ["is", "are", "was", "were", "do", "does", "did", "can", "will"]: return "yes/no"
    if first in ["what", "who", "where", "why", "when", "how", "which"]: return first
    return "other"