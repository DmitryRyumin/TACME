"""
Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Sequence-level feature aggregation for the TACME inference pipeline.

License: MIT License
"""

from typing import Final

import torch
from torch import Tensor

EXPECTED_AGGREGATED_LENGTH: Final = 2


class FeatureAggregationError(RuntimeError):
    """Raised when TACME feature aggregation fails."""


def mean_std_seq(
    features: Tensor,
) -> Tensor:
    """Aggregate [T, D] features into [2, D] as mean and population standard deviation."""

    if not isinstance(
        features,
        Tensor,
    ):
        raise FeatureAggregationError("Feature sequence must be a PyTorch tensor.")

    if features.ndim != 2:
        msg = f"Feature sequence must have shape [T, D]; got {tuple(features.shape)}."
        raise FeatureAggregationError(msg)

    if features.shape[0] <= 0:
        raise FeatureAggregationError("Feature sequence is empty.")

    if features.shape[1] <= 0:
        raise FeatureAggregationError("Feature dimension must be positive.")

    if features.dtype != torch.float32:
        msg = f"Feature sequence must use torch.float32; got {features.dtype}."
        raise FeatureAggregationError(msg)

    if not bool(
        torch.isfinite(
            features,
        )
        .all()
        .item()
    ):
        raise FeatureAggregationError("Feature sequence contains NaN or infinite values.")

    mean = features.mean(
        dim=0,
    )

    std = features.std(
        dim=0,
        correction=0,
    )

    aggregated = torch.stack(
        (
            mean,
            std,
        ),
        dim=0,
    )

    expected_shape = (
        EXPECTED_AGGREGATED_LENGTH,
        features.shape[1],
    )

    if (
        tuple(
            aggregated.shape,
        )
        != expected_shape
    ):
        msg = f"Unexpected aggregated feature shape: {tuple(aggregated.shape)}; expected {expected_shape}."
        raise FeatureAggregationError(msg)

    if aggregated.dtype != torch.float32:
        msg = f"Unexpected aggregated feature dtype: {aggregated.dtype}; expected torch.float32."
        raise FeatureAggregationError(msg)

    if aggregated.device != features.device:
        msg = f"Feature aggregation changed the tensor device: {features.device} -> {aggregated.device}."
        raise FeatureAggregationError(msg)

    if not bool(
        torch.isfinite(
            aggregated,
        )
        .all()
        .item()
    ):
        raise FeatureAggregationError("Aggregated features contain NaN or infinite values.")

    return aggregated.contiguous()


__all__ = [
    "EXPECTED_AGGREGATED_LENGTH",
    "FeatureAggregationError",
    "mean_std_seq",
]
