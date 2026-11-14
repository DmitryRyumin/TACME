"""
File: heads.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Task-specific TACME classification and regression heads.

License: MIT License
"""

import torch
from torch import Tensor, nn

SUPPORTED_HEAD_TYPES = frozenset(
    {
        "linear",
        "mlp",
    }
)


class TaskHead(nn.Module):
    """Task-specific prediction head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        dropout: float = 0.1,
        head_type: str = "mlp",
        apply_sigmoid: bool = False,
    ) -> None:
        super().__init__()

        normalized_head_type = head_type.lower()

        if normalized_head_type not in SUPPORTED_HEAD_TYPES:
            msg = f"Unknown head_type: {head_type!r}. Expected one of: linear, mlp."
            raise ValueError(msg)

        self.apply_sigmoid = apply_sigmoid

        if normalized_head_type == "linear":
            self.pre_output: nn.Module = nn.Identity()

            self.output = nn.Linear(
                input_dim,
                out_dim,
            )

        else:
            self.pre_output = nn.Sequential(
                nn.Linear(
                    input_dim,
                    hidden_dim,
                ),
                nn.LayerNorm(
                    hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(
                    dropout,
                ),
            )

            self.output = nn.Linear(
                hidden_dim,
                out_dim,
            )

    def forward_with_embeddings(
        self,
        x: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        """Return the task embedding and prediction."""

        embedding = self.pre_output(
            x,
        )

        prediction = self.output(
            embedding,
        )

        if self.apply_sigmoid:
            prediction = torch.sigmoid(
                prediction,
            )

        return (
            embedding,
            prediction,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """Return the task prediction."""

        _, prediction = self.forward_with_embeddings(
            x,
        )

        return prediction
