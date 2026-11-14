"""
File: experts.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Directed cross-modal experts and their stable specifications.

License: MIT License
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tacme.models.common.attention import TransformerEncoderLayer
from tacme.models.common.tensor_ops import masked_mean


class DirectedExpertBlock(nn.Module):
    """Directed expert in which one modality attends to another modality."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.layer = TransformerEncoderLayer(
            input_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            positional_encoding=False,
        )

    @staticmethod
    def _resample_sequence_to_length(
        x: Tensor,
        mask: Tensor | None,
        target_len: int,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        """Resample a masked sequence to a shared temporal length."""

        batch_size, _, dimension = x.shape

        output = x.new_zeros(
            (
                batch_size,
                target_len,
                dimension,
            )
        )

        output_mask = torch.ones(
            (
                batch_size,
                target_len,
            ),
            dtype=torch.bool,
            device=x.device,
        )

        for index in range(batch_size):
            current = x[index]

            valid_length = (
                current.shape[0]
                if mask is None
                else max(
                    int(mask[index].sum().item()),
                    1,
                )
            )

            current = current[:valid_length]

            current = current.transpose(
                0,
                1,
            ).unsqueeze(
                0,
            )

            current = F.interpolate(
                current,
                size=target_len,
                mode="linear",
                align_corners=False,
            )

            output[index] = current.squeeze(
                0,
            ).transpose(
                0,
                1,
            )

        return (
            output,
            output_mask,
        )

    def forward(
        self,
        query_seq: Tensor,
        context_seq: Tensor,
        *,
        query_mask: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        """Return the pooled directed cross-modal expert representation."""

        target_len = max(
            min(
                query_seq.shape[1],
                context_seq.shape[1],
            ),
            1,
        )

        (
            query,
            query_resampled_mask,
        ) = self._resample_sequence_to_length(
            query_seq,
            query_mask,
            target_len,
        )

        (
            context,
            context_resampled_mask,
        ) = self._resample_sequence_to_length(
            context_seq,
            context_mask,
            target_len,
        )

        key_padding_mask = ~context_resampled_mask

        x = self.layer(
            query,
            context,
            context,
            key_padding_mask=(key_padding_mask),
        )

        x = x + query

        return masked_mean(
            x,
            query_resampled_mask,
        )


@dataclass(
    frozen=True,
    slots=True,
)
class ExpertSpec:
    """Definition of one directed cross-modal expert."""

    query_mod: str
    context_mod: str

    @property
    def name(
        self,
    ) -> str:
        """Return the stable state-dict name for the expert."""

        return f"{self.query_mod}__to__{self.context_mod}"
