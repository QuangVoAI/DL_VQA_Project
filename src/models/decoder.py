"""Answer decoder with optional Bahdanau attention."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.data.dataset import PAD_IDX
from src.models.attention import BahdanauAttention
from src.models.encoder import CNNEncoder


class AnswerDecoder(nn.Module):
    """LSTM answer decoder with optional Bahdanau attention + padding mask.

    Input at each step: [embedding ; context_or_hidden ; image_feature].

    Args:
        vocab_size: Answer vocabulary size.
        embed_size: Embedding dimension.
        hidden_size: LSTM hidden dimension.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout rate.
        use_attention: Whether to use Bahdanau attention.
        pretrained_emb: Optional pre-trained embedding matrix.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_size: int = 300,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_attention: bool = False,
        pretrained_emb: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=PAD_IDX)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)

        if use_attention:
            self.attention = BahdanauAttention(hidden_size)

        lstm_in = embed_size + hidden_size + CNNEncoder.CNN_OUT_DIM
        self.lstm = nn.LSTM(
            lstm_in, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        token: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
        img_feat: torch.Tensor,
        q_outputs: Optional[torch.Tensor] = None,
        q_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode one time step.

        Args:
            token: Current input token ``(B,)``.
            hidden: LSTM hidden state ``(L, B, H)``.
            cell: LSTM cell state ``(L, B, H)``.
            img_feat: Image features ``(B, 512)``.
            q_outputs: Question encoder outputs ``(B, T, H)``.
            q_mask: Question padding mask ``(B, T)``.

        Returns:
            pred: Logits over answer vocabulary ``(B, V)``.
            hidden: Updated hidden state.
            cell: Updated cell state.
        """
        emb = self.embedding(token.unsqueeze(1))  # (B, 1, E)
        if self.use_attention and q_outputs is not None:
            ctx, _attn_w = self.attention(hidden[-1], q_outputs, q_mask)
            inp = torch.cat([emb, ctx.unsqueeze(1), img_feat.unsqueeze(1)], 2)
        else:
            inp = torch.cat([emb, hidden[-1].unsqueeze(1), img_feat.unsqueeze(1)], 2)

        out, (hidden, cell) = self.lstm(inp, (hidden, cell))
        pred = self.fc(self.fc_drop(out.squeeze(1)))  # (B, V)
        return pred, hidden, cell
