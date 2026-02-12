# Overview
This project is designed for Visual Question Answering (VQA) in Deep Learning. It integrates advanced machine learning models to process and analyze images and questions based on visual content.

# Project Structure
```
DL_VQA_Project/
│
├── data/                 # Contains dataset files
├── notebooks/            # Jupyter notebooks for experimentation
├── src/                 # Source code for the VQA model
│   ├── model.py          # Model architecture definitions
│   ├── train.py          # Training scripts
│   └── utils.py          # Utility functions
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

# Quick Start
1. Clone the repository:
   ```bash
   git clone https://github.com/QuangVoAI/DL_VQA_Project.git
   cd DL_VQA_Project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start training:
   ```bash
   python src/train.py
   ```

# Model Architecture
The VQA model is built using a combination of CNNs and LSTMs to extract features from images and questions respectively. The output is decoded to generate answers based on learned associations.

# Key Components
- **Data Loader:** Handles loading and preprocessing of the dataset.
- **Model Training:** Implements training loop with metrics logging.
- **Inference:** Functionality to make predictions based on new images and questions.

# Dependencies
- TensorFlow 2.x
- Keras
- NumPy
- pandas
- matplotlib
- scikit-learn

# Features
- Ability to train a model on custom datasets.
- Supports multiple question types.
- Visualization of training metrics.

# Device Support
The model can be trained or run on:
- CPU
- GPU (NVIDIA)

# Expected Results
The model is expected to achieve high accuracy on the validation set, with a focus on understanding the correlation between image content and questions.

# Next Steps
- Experiment with different model architectures
- Fine-tune hyperparameters
- Expand dataset with more images and question types

# Author
QuangVoAI, 2026-02-12 06:18:27 UTC