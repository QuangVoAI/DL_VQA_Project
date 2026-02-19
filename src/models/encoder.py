"""Image and Question encoders for VQA."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.data.dataset import PAD_IDX


class CNNEncoder(nn.Module):
    """Image feature extractor — scratch CNN or pretrained ResNet-18.

    Both paths output a 512-d feature vector per image.

    Args:
        pretrained: If True, use ResNet-18 with ImageNet weights
                    (freeze early layers, fine-tune last 10 params).
    """

    CNN_OUT_DIM: int = 512

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        if pretrained:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # → (B,512,1,1)
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
                nn.AdaptiveAvgPool2d((1, 1)),
            )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract image features.

        Args:
            images: ``(B, 3, 224, 224)``

        Returns:
            Image features ``(B, 512)``.
        """
        return self.cnn(images).flatten(1)


class QuestionEncoder(nn.Module):
    """Bi-directional LSTM question encoder with GloVe embeddings.

    Returns encoder outputs, final hidden states, and a padding mask.

    Args:
        vocab_size: Size of question vocabulary.
        embed_size: Embedding dimension (default 300 for GloVe).
        hidden_size: LSTM hidden dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate.
        pretrained_emb: Optional pre-trained embedding matrix.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 300,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_emb: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers,
            batch_first=True, bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        questions: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Encode question sequences.

        Args:
            questions: Padded question token IDs ``(B, T)``.
            lengths: True lengths per sample ``(B,)``.

        Returns:
            outputs: Encoder outputs ``(B, T, H)``.
            (h, c): Final LSTM states.
            mask: Padding mask ``(B, T)`` — 1 = real, 0 = pad.
        """
        mask = (questions != PAD_IDX).float()  # (B, T)
        emb = self.dropout(self.embedding(questions))  # (B, T, E)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, (h, c) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs, (h, c), mask
