"""GloVe embedding utilities."""

import os
import urllib.request
import zipfile
import numpy as np
import torch
from tqdm.auto import tqdm
from src.data.dataset import PAD_IDX

GLOVE_DIR = "data/glove"
GLOVE_FILE = os.path.join(GLOVE_DIR, "glove.6B.300d.txt")

def download_glove():
    if os.path.exists(GLOVE_FILE): return
    os.makedirs(GLOVE_DIR, exist_ok=True)
    url = "https://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = os.path.join(GLOVE_DIR, "glove.6B.zip")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract("glove.6B.300d.txt", GLOVE_DIR)
    os.remove(zip_path)

def load_glove_embeddings(vocab, embed_dim=300):
    glove = {}
    with open(GLOVE_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading GloVe"):
            parts = line.split()
            if len(parts) == embed_dim + 1:
                glove[parts[0]] = np.array(parts[1:], dtype=np.float32)

    matrix = np.random.normal(scale=0.6, size=(len(vocab), embed_dim)).astype(np.float32)
    matrix[PAD_IDX] = 0
    for word, idx in vocab.stoi.items():
        if word in glove: matrix[idx] = glove[word]
    return torch.from_numpy(matrix)