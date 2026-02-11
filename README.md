# DL_VQA_Project
Midterm Project of Quang and Thành

## Visual Question Answering with PyTorch

This project implements a Visual Question Answering (VQA) system using PyTorch, combining CNN-based image encoding with LSTM-based question processing.

## Project Structure

```
DL_VQA_Project/
├── data/                      # Dataset directory (excluded from git)
├── checkpoints/               # Model checkpoints (excluded from git)
├── src/                       # Source code
│   ├── dataset.py            # Vocabulary & VQADataset classes
│   ├── model.py              # EncoderCNN & DecoderRNN architectures
│   └── train.py              # Training loop and utilities
├── PROJECT_ROADMAP.md         # 2-week development plan
├── requirements.txt           # Python dependencies
└── .gitignore                 # Git ignore rules
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download VQA v2.0 dataset and place in `data/` directory

3. Configure paths in `src/train.py`

## Model Architecture

- **EncoderCNN**: ResNet-50 backbone for visual feature extraction
- **DecoderRNN**: LSTM-based question encoder and answer classifier
- **Answer Prediction**: Classification over top-K most common answers

## Usage

See `src/train.py` for training configuration and `PROJECT_ROADMAP.md` for detailed development plan.

## Features

- Modular, well-documented code with shape annotations
- Pretrained ResNet backbone with fine-tuning support
- Checkpoint saving/loading for training resumption
- Validation metrics and progress tracking
- Extensible architecture for attention mechanisms
