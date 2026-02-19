"""GloVe embedding download and loading utilities."""

from __future__ import annotations

import logging
import os
import urllib.request
import zipfile

import numpy as np
import torch
from tqdm.auto import tqdm

from src.data.dataset import Vocabulary, PAD_IDX

logger = logging.getLogger("VQA")

GLOVE_DIR = "data/glove"
GLOVE_FILE = os.path.join(GLOVE_DIR, "glove.6B.300d.txt")


def download_glove() -> None:
    """Download GloVe 6B 300d embeddings if not already present."""
    if os.path.exists(GLOVE_FILE):
        logger.info("GloVe 300d already downloaded.")
        return
    os.makedirs(GLOVE_DIR, exist_ok=True)
    url = "https://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = os.path.join(GLOVE_DIR, "glove.6B.zip")
    logger.info("Downloading GloVe 6B (~862 MB) ...")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract("glove.6B.300d.txt", GLOVE_DIR)
    os.remove(zip_path)
    logger.info("GloVe ready.")


def load_glove_embeddings(vocab: Vocabulary, embed_dim: int = 300) -> torch.Tensor:
    """Build embedding matrix from GloVe. OOV → N(0, 0.6), <PAD> → zeros.

    Args:
        vocab: Vocabulary instance with .stoi mapping.
        embed_dim: Embedding dimension (default 300).

    Returns:
        Tensor of shape (vocab_size, embed_dim).
    """
    logger.info("Loading GloVe vectors ...")
    glove: dict[str, np.ndarray] = {}
    with open(GLOVE_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading GloVe"):
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            if vec.shape[0] == embed_dim:
                glove[word] = vec

    matrix = np.random.normal(scale=0.6, size=(len(vocab), embed_dim)).astype(np.float32)
    matrix[PAD_IDX] = 0.0

    found = 0
    for word, idx in vocab.stoi.items():
        if word in glove:
            matrix[idx] = glove[word]
            found += 1

    logger.info(f"  GloVe coverage: {found}/{len(vocab)} ({found / len(vocab) * 100:.1f}%)")
    return torch.from_numpy(matrix)
