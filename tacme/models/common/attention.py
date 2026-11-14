"""
File: attention.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Checkpoint-compatible feed-forward, residual, positional, and attention layers.

License: MIT License
"""

import math
from typing import cast

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class PositionWiseFeedForward(nn.Module):
    """Position-wise feed-forward block used by TACME attention layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.layer_1 = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.layer_2 = nn.Linear(
            hidden_dim,
            input_dim,
        )

        self.dropout = nn.Dropout(
            dropout,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Apply the feed-forward transformation."""

        x = self.layer_1(
            x,
        )

        x = F.gelu(
            x,
        )

        x = self.dropout(
            x,
        )

        return cast(
            Tensor,
            self.layer_2(
                x,
            ),
        )


class AddAndNorm(nn.Module):
    """Checkpoint-compatible TACME residual and normalization block."""

    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(
            input_dim,
        )

        self.dropout = nn.Dropout(
            dropout,
        )

    def forward(
        self,
        x: Tensor,
        residual: Tensor,
    ) -> Tensor:
        """Apply residual addition and layer normalization."""

        return cast(
            Tensor,
            self.norm(
                x
                + self.dropout(
                    residual,
                )
            ),
        )


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    pe: Tensor

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()

        self.dropout = nn.Dropout(
            p=dropout,
        )

        position = torch.arange(
            max_len,
        ).unsqueeze(
            1,
        )

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
            )
            * (
                -math.log(
                    10000.0,
                )
                / d_model
            )
        )

        pe = torch.zeros(
            max_len,
            d_model,
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer(
            "pe",
            pe,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Add positional information to a sequence."""

        positional = self.pe[
            : x.size(
                1,
            )
        ].detach()

        return cast(
            Tensor,
            self.dropout(x + positional),
        )


class TransformerEncoderLayer(nn.Module):
    """Checkpoint-compatible TACME multi-head attention encoder layer."""

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        positional_encoding: bool = True,
    ) -> None:
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            input_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.feed_forward = PositionWiseFeedForward(
            input_dim,
            input_dim,
            dropout=dropout,
        )

        self.add_norm_after_attention = AddAndNorm(
            input_dim,
            dropout=dropout,
        )

        self.add_norm_after_ff = AddAndNorm(
            input_dim,
            dropout=dropout,
        )

        self.positional_encoding = (
            PositionalEncoding(
                input_dim,
            )
            if positional_encoding
            else None
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply attention and feed-forward transformations."""

        if self.positional_encoding is not None:
            key = self.positional_encoding(
                key,
            )

            value = self.positional_encoding(
                value,
            )

            query = self.positional_encoding(
                query,
            )

        attention_output, _ = self.self_attention(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        x = self.add_norm_after_attention(
            attention_output,
            query,
        )

        feed_forward_output = self.feed_forward(
            x,
        )

        return cast(
            Tensor,
            self.add_norm_after_ff(
                feed_forward_output,
                x,
            ),
        )
