# Visual Question Answering (VQA) Project
Midterm Project of Quang and Thành

## Overview
This project implements a Visual Question Answering (VQA) system using a CNN-LSTM architecture. The system takes an image and a natural language question as input and generates a natural language answer.

## Architecture
- **Encoder (CNN)**: Uses a pretrained ResNet model to extract visual features from images
- **Decoder (RNN)**: Uses LSTM to process questions and generate answers based on image features

## Project Structure
```
DL_VQA_Project/
├── data/                  # Dataset storage (ignored by git)
├── src/                   # Source code
│   ├── dataset.py        # Dataset and vocabulary classes
│   ├── model.py          # CNN-LSTM model architecture
│   ├── train.py          # Training loop
│   └── utils.py          # Utility functions
├── checkpoints/          # Model weights (ignored by git)
├── notebooks/            # Jupyter notebooks for experimentation
├── requirements.txt      # Python dependencies
└── PROJECT_ROADMAP.md   # Development timeline
```

## Installation
```bash
# Clone the repository
git clone https://github.com/QuangVoAI/DL_VQA_Project.git
cd DL_VQA_Project

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for tokenization)
python -c "import nltk; nltk.download('punkt')"
```

## Usage
### Training
```bash
python src/train.py --data_path data/ --checkpoint_dir checkpoints/
```

### Dataset Preparation
Place your VQA dataset in the `data/` directory. The expected format includes:
- Images (`.jpg`, `.png`)
- Questions file (JSON format)
- Answers file (JSON format)

## Requirements
- Python >= 3.8
- PyTorch >= 2.0.0
- See `requirements.txt` for full list of dependencies

## Development Timeline
See [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) for the detailed development plan.

## Team
- Quang
- Thành
