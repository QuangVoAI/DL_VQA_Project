"""Helper utilities: device management, sequence decoding, seeding, and logging."""

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
    """
    Tự động phát hiện thiết bị tốt nhất hiện có: CUDA (Nvidia) → MPS (Mac M1/M2/M3) → CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """
    Thiết lập seed ngẫu nhiên để đảm bảo tính tái lập kết quả trên tất cả các thư viện.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Thiết lập hệ thống ghi log (logging) ra cả file và terminal.
    """
    os.makedirs(log_dir, exist_ok=True)
    vqa_logger = logging.getLogger("VQA")
    vqa_logger.setLevel(logging.INFO)

    # Avoid creating multiple handlers if function is called multiple times
    if not vqa_logger.handlers:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"vqa_{timestamp}.log")
        
        # Log to file
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        vqa_logger.addHandler(fh)
        
        # Log to console
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        vqa_logger.addHandler(ch)

    return vqa_logger


def decode_sequence(sequence: list[int], vocab: Any) -> str:
    """
    Chuyển đổi danh sách các token ID thành chuỗi văn bản.
    Dừng lại khi gặp token <EOS> và bỏ qua các token <PAD>/<SOS>.
    """
    tokens = []
    for idx in sequence:
        # Stop decoding upon encountering sentence end token
        if idx == EOS_IDX:
            break
        # Take only actual words (exclude PAD or SOS)
        if idx not in (PAD_IDX, SOS_IDX):
            word = vocab.itos.get(idx, "<UNK>")
            tokens.append(word)
    
    return " ".join(tokens)