"""
File: flow_matching.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Conditional flow-matching velocity network.

License: MIT License
"""

import math
from typing import cast

import torch
from torch import Tensor, nn


class FlowMatchingMLP(nn.Module):
    """Conditional velocity field used by TACME flow matching."""

    def __init__(
        self,
        state_dim: int,
        cond_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                state_dim + cond_dim + 3,
                hidden_dim,
            ),
            nn.LayerNorm(
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.LayerNorm(
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                hidden_dim,
                state_dim,
            ),
        )

    def forward(
        self,
        x_t: Tensor,
        cond: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Estimate the conditional flow velocity."""

        if t.ndim == 1:
            t = t.unsqueeze(
                -1,
            )

        t_embed = torch.cat(
            [
                t,
                torch.sin(2 * math.pi * t),
                torch.cos(2 * math.pi * t),
            ],
            dim=-1,
        )

        return cast(
            Tensor,
            self.net(
                torch.cat(
                    [
                        x_t,
                        cond,
                        t_embed,
                    ],
                    dim=-1,
                )
            ),
        )
