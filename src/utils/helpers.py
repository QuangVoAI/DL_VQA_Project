"""Helper utilities: device, decode, seed, etc."""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime
from typing import Any

import numpy as np
import torch

from src.data.dataset import PAD_IDX, SOS_IDX, EOS_IDX

logger = logging.getLogger("VQA")


def get_device() -> torch.device:
    """Auto-detect best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Set up logging to both file and console.

    Args:
        log_dir: Directory for log files.

    Returns:
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    vqa_logger = logging.getLogger("VQA")
    vqa_logger.setLevel(logging.INFO)

    if not vqa_logger.handlers:
        fh = logging.FileHandler(f"{log_dir}/vqa_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        vqa_logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        vqa_logger.addHandler(ch)

    return vqa_logger


def decode_sequence(sequence: list[int], vocab: Any) -> str:
    """Convert a list of token IDs back to a text string.

    Stops at <EOS> and skips <PAD>/<SOS> tokens.

    Args:
        sequence: List of integer token IDs.
        vocab: Vocabulary with itos mapping.

    Returns:
        Decoded text string.
    """
    tokens = []
    for idx in sequence:
        if idx == vocab.stoi["<EOS>"]:
            break
        if idx not in (vocab.stoi["<PAD>"], vocab.stoi["<SOS>"]):
            tokens.append(vocab.itos.get(idx, "<UNK>"))
    return " ".join(tokens)
