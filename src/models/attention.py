"""Bahdanau (Additive) Attention with optional padding mask."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    """Additive attention mechanism (Bahdanau et al., 2015).

    Computes: score = V^T tanh(W_a · query + U_a · keys)
    Optionally masks padding positions to -inf before softmax.

    Args:
        hidden_size: Dimensionality of query and key vectors.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention context and weights.

        Args:
            query: Decoder hidden state ``(B, H)``.
            keys: Encoder outputs ``(B, T, H)``.
            mask: Padding mask ``(B, T)`` — 1 = real token, 0 = pad.

        Returns:
            context: Weighted sum of keys ``(B, H)``.
            weights: Attention distribution ``(B, T)``.
        """
        scores = self.Va(torch.tanh(
            self.Wa(query.unsqueeze(1)) + self.Ua(keys)
        )).squeeze(2)  # (B, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(scores, dim=1)  # (B, T)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)  # (B, H)
        return context, weights
