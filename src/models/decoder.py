"""Answer decoder with Spatial Attention + Bahdanau text attention."""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from src.data.dataset import PAD_IDX
from src.models.attention import BahdanauAttention, SpatialAttention
from src.models.encoder import CNNEncoder

class AnswerDecoder(nn.Module):
    def __init__(
        self, vocab_size: int, embed_size: int = 300, hidden_size: int = 512,
        num_layers: int = 2, dropout: float = 0.3, use_attention: bool = False,
        pretrained_emb: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)

        if use_attention:
            self.text_attention = BahdanauAttention(hidden_size)
            self.spatial_attention = SpatialAttention(hidden_size, CNNEncoder.CNN_OUT_DIM)

        lstm_in = embed_size + hidden_size + CNNEncoder.CNN_OUT_DIM
        self.lstm = nn.LSTM(lstm_in, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.res_proj = nn.Linear(lstm_in, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, token: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor,
        img_feat: torch.Tensor, q_outputs: Optional[torch.Tensor] = None, q_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.embedding(token.unsqueeze(1))
        if self.use_attention and q_outputs is not None:
            text_ctx, _ = self.text_attention(hidden[-1], q_outputs, q_mask)
            img_ctx, _ = self.spatial_attention(hidden[-1], img_feat)
            inp = torch.cat([emb, text_ctx.unsqueeze(1), img_ctx.unsqueeze(1)], 2)
        else:
            img_ctx = img_feat.mean(dim=1)
            inp = torch.cat([emb, hidden[-1].unsqueeze(1), img_ctx.unsqueeze(1)], 2)

        out, (hidden, cell) = self.lstm(inp, (hidden, cell))
        residual = self.res_proj(inp.squeeze(1))
        out_res = self.layer_norm(out.squeeze(1) + residual)
        pred = self.fc(self.fc_drop(out_res))
        return pred, hidden, cell