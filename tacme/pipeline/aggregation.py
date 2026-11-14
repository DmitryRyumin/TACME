"""
File: aggregation.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Aggregating and validating unimodal feature tensors.

License: MIT License
"""

import torch
from torch import Tensor

from tacme.extractors.aggregation import mean_std_seq
from tacme.pipeline.contracts import AGGREGATED_SEQUENCE_LENGTH, PipelineError


def _aggregate_and_batch(
    modality: str,
    raw_features: Tensor,
    expected_dim: int,
) -> Tensor:
    """Apply mean+STD aggregation and add the explicit batch dimension."""

    aggregated = mean_std_seq(
        raw_features,
    )

    expected_shape = (
        AGGREGATED_SEQUENCE_LENGTH,
        expected_dim,
    )

    if (
        tuple(
            aggregated.shape,
        )
        != expected_shape
    ):
        msg = f"Unexpected aggregated {modality!r} feature shape: {tuple(aggregated.shape)}; expected {expected_shape}."
        raise PipelineError(msg)

    if aggregated.dtype != torch.float32:
        msg = f"Unexpected aggregated {modality!r} feature dtype: {aggregated.dtype}; expected torch.float32."
        raise PipelineError(msg)

    if not bool(
        torch.isfinite(
            aggregated,
        )
        .all()
        .item()
    ):
        msg = f"Aggregated {modality!r} features contain NaN or infinite values."
        raise PipelineError(msg)

    return aggregated.unsqueeze(
        0,
    ).contiguous()
