"""Evaluation pipeline with question-type breakdown."""

from __future__ import annotations
import logging
import os
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.dataset import PAD_IDX
from src.data.preprocessing import majority_answer, classify_question
from src.utils.helpers import decode_sequence
from src.utils.metrics import batch_metrics, compute_exact_match, compute_f1, compute_meteor

logger = logging.getLogger("VQA")


def _safe_load_checkpoint(ckpt_path: str, device: torch.device) -> dict | None:
    """Tải checkpoint, thử weights_only=True trước, fallback sang False nếu lỗi."""
    if not os.path.exists(ckpt_path):
        return None
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception:
        # Fallback cho checkpoint cũ hoặc PyTorch < 2.0
        return torch.load(ckpt_path, map_location=device)


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
    """Chạy inference trên test set, tải checkpoint tốt nhất nếu có."""

    ckpt_path = os.path.join(ckpt_dir, f"best_{name}.pth")
    ckpt = _safe_load_checkpoint(ckpt_path, device)
    if ckpt is not None:
        # Hỗ trợ checkpoint lưu dict {"model": state_dict} hoặc state_dict thẳng
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning(f"  [!] Missing keys for {name}: {missing}")
        if unexpected:
            logger.warning(f"  [!] Unexpected keys for {name}: {unexpected}")
        logger.info(f"  Loaded checkpoint: {ckpt_path}")
    else:
        logger.warning(f"  No checkpoint found at {ckpt_path}, using current weights.")

    model.eval()
    preds: list[str] = []
    refs: list[str]  = []
    questions_text: list[str] = []

    with torch.no_grad():
        for imgs, qs, ql, ans, al, ans_txt in tqdm(test_loader, desc=f"Test {name}"):
            imgs, qs, ql = imgs.to(device), qs.to(device), ql.to(device)
            gen = model.generate(imgs, qs, ql, use_beam=True, beam_width=beam_width)
            for i in range(gen.size(0)):
                preds.append(decode_sequence(gen[i].cpu().tolist(), answer_vocab))
                refs.append(ans_txt[i])
                questions_text.append(decode_sequence(qs[i].cpu().tolist(), question_vocab))

    m = batch_metrics(preds, refs)
    logger.info(
        f"  {name} F1={m['f1']:.4f} METEOR={m['meteor']:.4f} "
        f"ROUGE-L={m['rouge_l']:.4f} B4={m['bleu4']:.4f}"
    )

    return {"metrics": m, "preds": preds, "refs": refs, "questions": questions_text}


def evaluate_by_question_type(
    preds: list[str], refs: list, questions: list[str]
) -> dict[str, dict[str, Any]]:
    """Phân nhóm kết quả theo loại câu hỏi và tính EM, F1 từng nhóm."""
    type_data: dict[str, dict] = defaultdict(lambda: {"preds": [], "refs": []})
    for p, r, q in zip(preds, refs, questions):
        qtype = classify_question(q)
        type_data[qtype]["preds"].append(p)
        type_data[qtype]["refs"].append(r if isinstance(r, str) else majority_answer(r))

    results: dict[str, dict] = {}
    for qtype, data in sorted(type_data.items(), key=lambda x: -len(x[1]["preds"])):
        ems = [compute_exact_match(p, r) for p, r in zip(data["preds"], data["refs"])]
        f1s = [compute_f1(p, r)          for p, r in zip(data["preds"], data["refs"])]
        results[qtype] = {
            "total": len(data["preds"]),
            "em": float(np.mean(ems)),
            "f1": float(np.mean(f1s)),
        }
    return results


def get_failure_cases(
    preds: list[str], refs: list, questions: list[str], n: int = 20
) -> list[dict[str, Any]]:
    """Trả về n mẫu dự đoán sai nhất (F1 thấp nhất)."""
    failures = []
    for p, r, q in zip(preds, refs, questions):
        ref_str = r if isinstance(r, str) else majority_answer(r)
        failures.append({
            "question":   q,
            "prediction": p,
            "reference":  ref_str,
            "f1":         compute_f1(p, ref_str),
            "type":       classify_question(q),
        })
    failures.sort(key=lambda x: x["f1"])
    return failures[:n]