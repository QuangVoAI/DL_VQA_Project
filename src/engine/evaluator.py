"""Evaluation pipeline with question-type breakdown and error analysis."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import PAD_IDX, SOS_IDX, EOS_IDX
from src.data.preprocessing import normalize_answer, majority_answer, classify_question
from src.utils.helpers import decode_sequence
from src.utils.metrics import batch_metrics, compute_exact_match, compute_f1, compute_meteor

logger = logging.getLogger("VQA")


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    answer_vocab: Any,
    question_vocab: Any,
    device: torch.device,
    ckpt_dir: str = "checkpoints",
    name: str = "model",
    beam_width: int = 5,
) -> dict[str, Any]:
    """Evaluate a model on the test set using beam search.

    Args:
        model: VQAModel in eval mode.
        test_loader: Test DataLoader.
        answer_vocab: Answer vocabulary for decoding.
        question_vocab: Question vocabulary for decoding.
        device: Torch device.
        ckpt_dir: Directory containing checkpoints.
        name: Model variant name.
        beam_width: Beam search width.

    Returns:
        Dict with 'metrics', 'preds', 'refs', 'questions'.
    """
    # Load best checkpoint if available
    ckpt_path = os.path.join(ckpt_dir, f"best_{name}.pth")
    if os.path.exists(ckpt_path):
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)["model"]
        )
        logger.info(f"Loaded best checkpoint for {name}")

    model.eval()
    preds, refs, questions_text = [], [], []

    with torch.no_grad():
        for imgs, qs, ql, ans, al, ans_txt in tqdm(test_loader, desc=f"Test {name}"):
            imgs = imgs.to(device)
            qs = qs.to(device)
            ql = ql.to(device)
            gen = model.generate(imgs, qs, ql, use_beam=True, beam_width=beam_width)
            for i in range(gen.size(0)):
                preds.append(decode_sequence(gen[i].cpu().tolist(), answer_vocab))
                refs.append(ans_txt[i])
                questions_text.append(decode_sequence(qs[i].cpu().tolist(), question_vocab))

    m = batch_metrics(preds, refs)

    logger.info(
        f"  {name}  Acc={m['accuracy']:.4f}  EM={m['em']:.4f}  "
        f"F1={m['f1']:.4f}  METEOR={m['meteor']:.4f}  B4={m['bleu4']:.4f}"
    )

    return {
        "metrics": m,
        "preds": preds,
        "refs": refs,
        "questions": questions_text,
    }


def evaluate_by_question_type(
    preds: list[str],
    refs: list[str],
    questions: list[str],
) -> dict[str, dict[str, float]]:
    """Break down evaluation metrics by question type.

    Uses ``classify_question()`` to categorize each question into:
    yes/no, counting, what, who, where, why, when, how, which, other.

    Args:
        preds: Predicted answer strings.
        refs: Reference answer strings.
        questions: Question strings.

    Returns:
        Dict mapping question_type → {'total': int, 'em': float, 'f1': float, 'meteor': float}.
    """
    type_data: dict[str, dict[str, list]] = defaultdict(lambda: {"preds": [], "refs": []})

    for p, r, q in zip(preds, refs, questions):
        qtype = classify_question(q)
        type_data[qtype]["preds"].append(p)
        ref_str = r if isinstance(r, str) else majority_answer(r)
        type_data[qtype]["refs"].append(ref_str)

    results: dict[str, dict[str, float]] = {}
    for qtype, data in sorted(type_data.items(), key=lambda x: -len(x[1]["preds"])):
        n = len(data["preds"])
        ems = [compute_exact_match(p, r) for p, r in zip(data["preds"], data["refs"])]
        f1s = [compute_f1(p, r) for p, r in zip(data["preds"], data["refs"])]
        meteors = [compute_meteor(p, r) for p, r in zip(data["preds"], data["refs"])]

        import numpy as np
        results[qtype] = {
            "total": n,
            "em": float(np.mean(ems)),
            "f1": float(np.mean(f1s)),
            "meteor": float(np.mean(meteors)),
        }

    return results


def get_failure_cases(
    preds: list[str],
    refs: list[str],
    questions: list[str],
    n: int = 20,
) -> list[dict[str, str]]:
    """Get the worst prediction failures (lowest F1 scores).

    Args:
        preds: Predicted answers.
        refs: Reference answers.
        questions: Questions.
        n: Number of worst cases to return.

    Returns:
        List of dicts with 'question', 'prediction', 'reference', 'f1', 'type'.
    """
    failures = []
    for p, r, q in zip(preds, refs, questions):
        ref_str = r if isinstance(r, str) else majority_answer(r)
        f1 = compute_f1(p, ref_str)
        failures.append({
            "question": q,
            "prediction": p,
            "reference": ref_str,
            "f1": f1,
            "type": classify_question(q),
        })

    failures.sort(key=lambda x: x["f1"])
    return failures[:n]
