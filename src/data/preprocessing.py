"""Text preprocessing utilities for VQA."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

# ── Contraction mapping ──
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
    """Pick the most common annotator answer (A-OKVQA has 10 per question)."""
    if not direct_answers:
        return ""
    normed = [normalize_answer(a) for a in direct_answers]
    counter = Counter(normed)
    return counter.most_common(1)[0][0]


def best_rationale(rationales: list[str]) -> str:
    """Pick the longest rationale as the generation target."""
    if not rationales:
        return ""
    return max(rationales, key=lambda r: len(r.split()))


def extract_answer(item: dict[str, Any]) -> str:
    """Extract the best available answer from a dataset item.

    Priority: _target_answer → rationale → direct_answers → MCQ choice.
    """
    if "_target_answer" in item:
        return item["_target_answer"]
    rationales = item.get("rationales", [])
    if rationales:
        return best_rationale(rationales)
    direct_answers = item.get("direct_answers", [])
    if direct_answers:
        return majority_answer(direct_answers)
    choices = item.get("choices", [])
    correct_idx = item.get("correct_choice_idx", None)
    if choices and correct_idx is not None:
        return choices[correct_idx]
    return ""


def expand_data_with_rationales(data_list: list[dict]) -> list[dict]:
    """Expand training data by using ALL rationales (~3x).

    A-OKVQA has 3 rationales per question — each becomes a separate sample.
    """
    expanded: list[dict] = []
    for item in data_list:
        rationales = item.get("rationales", [])
        if rationales and len(rationales) > 1:
            for rat in rationales:
                new_item = dict(item)
                new_item["_target_answer"] = rat
                expanded.append(new_item)
        else:
            expanded.append(item)
    return expanded


# ── Question‐type classification ──
_YES_NO_WORDS = {"is", "are", "was", "were", "do", "does", "did", "can",
                 "could", "will", "would", "has", "have", "had", "should"}
_COUNTING_WORDS = {"how many", "how much", "count", "number of"}
_LOCATION_WORDS = {"where"}
_REASON_WORDS = {"why"}
_TIME_WORDS = {"when"}


def classify_question(question: str) -> str:
    """Classify a question into one of: yes/no, counting, what, who, where, why, when, how, other.

    Uses first-word heuristic plus bigram matching for counting questions.
    """
    q = question.lower().strip()
    words = q.split()
    if not words:
        return "other"

    first = words[0]
    bigram = " ".join(words[:2]) if len(words) >= 2 else ""

    # Counting (check bigram first)
    if any(kw in q for kw in _COUNTING_WORDS):
        return "counting"
    # Yes/No
    if first in _YES_NO_WORDS:
        return "yes/no"
    # WH-questions
    if first == "what":
        return "what"
    if first == "who" or first == "whose" or first == "whom":
        return "who"
    if first in _LOCATION_WORDS:
        return "where"
    if first in _REASON_WORDS:
        return "why"
    if first in _TIME_WORDS:
        return "when"
    if first == "how":
        return "how"
    if first == "which":
        return "which"

    return "other"
