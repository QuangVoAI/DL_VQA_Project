"""
VQA Project Source Package

This package contains the core components for Visual Question Answering:
- dataset: Data loading and vocabulary management
- model: CNN-LSTM architecture
- train: Training loop and utilities
- utils: Helper functions
"""

from .dataset import Vocabulary, VQADataset
from .model import EncoderCNN, DecoderRNN, VQAModel

__all__ = [
    'Vocabulary',
    'VQADataset',
    'EncoderCNN',
    'DecoderRNN',
    'VQAModel'
]

__version__ = '0.1.0'
