# DL_VQA_Project — A-OKVQA

**Midterm Project of XuanQuangVo (523H0173) and XuanThanhHoang (523H0178)**

A Visual Question Answering (VQA) system built with PyTorch on the **A-OKVQA** dataset (COCO images). The model takes an image + question as input and **generates a natural-language answer** via LSTM decoder.

## Project Structure

```
DL_VQA_Project/
├── VQA.ipynb                    # Main notebook (imports from src/)
├── configs/
│   └── default.yaml             # All hyperparameters in one place
├── src/                         # Modular Python package
│   ├── config.py                # Dataclass-based YAML config system
│   ├── data/
│   │   ├── preprocessing.py     # Text normalization, question classification
│   │   ├── dataset.py           # Vocabulary, VQADataset, collate_fn
│   │   └── glove.py             # GloVe download & embedding loading
│   ├── models/
│   │   ├── attention.py         # Bahdanau (Additive) Attention
│   │   ├── encoder.py           # CNNEncoder + QuestionEncoder (LSTM)
│   │   ├── decoder.py           # AnswerDecoder (LSTM + optional attention)
│   │   └── vqa_model.py         # Full VQAModel (forward + generate + beam search)
│   ├── engine/
│   │   ├── trainer.py           # Training loop, early stopping, logging
│   │   └── evaluator.py         # Evaluation, question-type breakdown, failure analysis
│   └── utils/
│       ├── metrics.py           # 8 metrics: VQA Acc, EM, F1, METEOR, BLEU-1/2/3/4
│       ├── helpers.py           # Seeds, device, logging, decode_sequence
│       └── visualization.py     # 7 plot functions: curves, radar, bar, attention, confusion
├── scripts/                     # CLI tools
│   ├── train.py                 # Train models from command line
│   ├── evaluate.py              # Evaluate checkpoints with analysis
│   ├── inference.py             # Inference pipeline + ONNX export
│   └── multi_seed_train.py      # Multi-seed training + paired t-test
├── checkpoints/                 # Saved model weights
├── data/processed/              # Cached vocabularies
├── requirements.txt
└── README.md
```

## Overview

| Component | Detail |
|-----------|--------|
| **Dataset** | A-OKVQA — ~17K train (3x expanded → ~51K), ~1.1K test |
| **Image Encoder** | CNN from scratch / Pretrained ResNet-18 |
| **Question Encoder** | LSTM 2-layer, GloVe 300d, dropout=0.3 |
| **Answer Decoder** | LSTM 2-layer, GloVe 300d, dropout=0.3 (generative) |
| **Attention** | Bahdanau (Additive) with padding mask |
| **Training** | Label Smoothing (0.1) · Cosine Annealing · Early Stopping (patience=5) |
| **Decoding** | Greedy (train) / Beam Search w=5 + length penalty (eval) |
| **Config** | YAML-based (`configs/default.yaml`) — single source of truth |
| **Evaluation** | 8 metrics · question-type breakdown · confusion matrix · failure analysis |

### 4 Model Variants

| # | CNN | Attention | Description |
|---|-----|-----------|-------------|
| M1 | Scratch | ✗ | Baseline |
| M2 | Scratch | ✓ | + Attention |
| M3 | Pretrained | ✗ | + Transfer learning |
| M4 | Pretrained | ✓ | **Full model** |

### Pipeline

```
COCO Image → Resize 224 → Normalize → CNN Encoder → (B, 512)
Question   → GloVe 300d → LSTM Encoder (2L) → (B, T, 512) + padding mask
                                ↓
              LSTM Decoder (2L) + Bahdanau Attention → token-by-token generation
                                ↓
              Beam Search (w=5, length penalty α=0.6) → answer text
```

### Evaluation Metrics (8 total)

| Metric | Description |
|--------|-------------|
| **VQA Accuracy** | Soft accuracy: min(#annotators_agree / 3, 1.0) |
| **Exact Match (EM)** | 1 if normalized pred == normalized ref |
| **Token F1** | Precision/Recall over word tokens |
| **METEOR** | Synonym + stemming aware overlap |
| **BLEU-1/2/3/4** | N-gram precision with brevity penalty (method4 smoothing) |

### Key Features

- **Modular codebase** — all code in `src/` with type hints and docstrings
- **YAML config** — one file controls all hyperparameters
- **Attention padding mask** — no information leak to `<PAD>` tokens
- **Beam search** with Wu et al. (2016) length normalization
- **Early stopping** (patience=5) with best-model checkpointing
- **Question-type analysis** — EM/F1/METEOR breakdown by type (what, who, counting, ...)
- **Confusion matrix** — correct/incorrect by question type
- **Failure analysis** — top-K worst predictions for error debugging
- **Attention overlay** — 3-panel visualization (image + token importance + word cloud)
- **Multi-seed training** with paired t-test for statistical significance
- **ONNX export** + batch inference for production deployment
- **Full reproducibility** (deterministic seeds, cudnn.deterministic)

## How to Run

### Notebook (recommended)

```bash
pip install -r requirements.txt
# Open VQA.ipynb and run all cells — dataset + GloVe download automatically
```

### CLI Training

```bash
# Train all 4 model variants
python scripts/train.py --config configs/default.yaml

# Train specific models with overrides
python scripts/train.py --models M1_Scratch_NoAttn M4_Pretrained_Attn --epochs 30 --lr 0.001

# Multi-seed training with significance testing
python scripts/multi_seed_train.py --seeds 42 123 456 --ttest
```

### CLI Evaluation

```bash
# Evaluate with question-type and failure analysis
python scripts/evaluate.py --config configs/default.yaml --question-type-analysis --failure-analysis
```

### CLI Inference

```bash
# Single prediction
python scripts/inference.py --image photo.jpg --question "What color is the car?"

# Batch from JSON
python scripts/inference.py --batch questions.json --output results.json

# ONNX export
python scripts/inference.py --export-onnx model.onnx
```

### Python API

```python
from scripts.inference import VQAInferencePipeline

pipe = VQAInferencePipeline(
    "checkpoints/best_M4_Pretrained_Attn.pth",
    "data/processed/vocab_aokvqa.pth"
)
answer = pipe.predict(image, "What color is the cat?")
```
