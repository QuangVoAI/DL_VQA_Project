"""Image and Question encoders for VQA."""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.data.dataset import PAD_IDX

class CNNEncoder(nn.Module):
    """Image feature extractor. Outputs a spatial grid instead of a flat vector."""
    CNN_OUT_DIM: int = 512

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # Bỏ lớp AdaptiveAvgPool2d và FC để giữ nguyên spatial map (7x7)
            self.cnn = nn.Sequential(*list(resnet.children())[:-2]) 
            for p in list(self.cnn.parameters())[:-10]:
                p.requires_grad = False
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(True),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.AdaptiveAvgPool2d((7, 7)), # Output lưới 7x7
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args: images: (B, 3, 224, 224)
        Returns: Image features (B, 49, 512) representing 49 spatial regions.
        """
        features = self.cnn(images)  # (B, 512, 7, 7)
        B, C, H, W = features.size()
        # Chuyển đổi thành (B, 49, 512)
        return features.view(B, C, H * W).permute(0, 2, 1)

class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int = 300, hidden_size: int = 512, num_layers: int = 2, dropout: float = 0.3, pretrained_emb: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=False, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, questions: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        mask = (questions != PAD_IDX).float()
        emb = self.dropout(self.embedding(questions))
        cpu_lengths = lengths.detach().cpu().to(torch.int64).clamp(min=1)
        packed = pack_padded_sequence(emb, cpu_lengths, batch_first=True, enforce_sorted=False)
        outputs, (h, c) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs, (h, c), mask