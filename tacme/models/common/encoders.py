"""
File: encoders.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Temporal encoding of input modality sequences.

License: MIT License
"""

from torch import Tensor, nn

from tacme.models.common.attention import TransformerEncoderLayer
from tacme.models.common.tensor_ops import ensure_sequence_input


class TemporalModalityEncoder(nn.Module):
    """Project and temporally encode one modality sequence."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        tr_layers: int = 5,
        dropout: float = 0.1,
        positional_encoding: bool = True,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.LayerNorm(
                hidden_dim,
            ),
            nn.Dropout(
                dropout,
            ),
        )

        self.transformer = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    input_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    positional_encoding=(positional_encoding),
                )
                for _ in range(tr_layers)
            ]
        )

    def forward(
        self,
        features: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        """Encode a modality sequence and return its normalized mask."""

        features, mask = ensure_sequence_input(
            features,
            mask,
        )

        x = self.input_proj(
            features,
        )

        key_padding_mask = ~mask

        for layer in self.transformer:
            attention = layer(
                x,
                x,
                x,
                key_padding_mask=(key_padding_mask),
            )

            x = x + attention

        return (
            x,
            mask,
        )
