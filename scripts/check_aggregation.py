"""
Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Runtime smoke test for TACME mean-and-standard-deviation feature aggregation.

License: MIT License
"""

import argparse
import sys
from typing import Final

import torch
from torch import Tensor

from tacme.extractors import (
    EXPECTED_AGGREGATED_LENGTH,
    FeatureAggregationError,
    mean_std_seq,
)
from tacme.models.common import resolve_device

TEST_SEQUENCE_LENGTH: Final = 3
TEST_FEATURE_DIM: Final = 4

ABSOLUTE_TOLERANCE: Final = 1e-6
RELATIVE_TOLERANCE: Final = 1e-6


def _parse_arguments() -> argparse.Namespace:
    """Parse aggregation check arguments."""

    parser = argparse.ArgumentParser(
        description=("Verify TACME mean_std_seq feature aggregation on deterministic feature sequences."),
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device used for the smoke test. Default: auto",
    )

    return parser.parse_args()


def _create_test_features(
    device: torch.device,
) -> Tensor:
    """Create one deterministic [T, D] feature sequence."""

    return torch.tensor(
        [
            [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            [
                3.0,
                6.0,
                9.0,
                12.0,
            ],
            [
                5.0,
                10.0,
                15.0,
                20.0,
            ],
        ],
        dtype=torch.float32,
        device=device,
    )


def _validate_aggregation(
    source: Tensor,
    aggregated: Tensor,
) -> None:
    """Validate aggregated shape, dtype, device, and values."""

    expected_shape = (
        EXPECTED_AGGREGATED_LENGTH,
        source.shape[1],
    )

    if (
        tuple(
            aggregated.shape,
        )
        != expected_shape
    ):
        msg = f"Unexpected aggregated feature shape: {tuple(aggregated.shape)}; expected {expected_shape}."
        raise RuntimeError(msg)

    if aggregated.dtype != torch.float32:
        msg = f"Unexpected aggregated feature dtype: {aggregated.dtype}; expected torch.float32."
        raise RuntimeError(msg)

    if aggregated.device != source.device:
        msg = f"Unexpected aggregated feature device: {aggregated.device}; expected {source.device}."
        raise RuntimeError(msg)

    if not bool(
        torch.isfinite(
            aggregated,
        )
        .all()
        .item()
    ):
        raise RuntimeError("Aggregated features contain NaN or infinite values.")

    expected_mean = source.mean(
        dim=0,
    )

    expected_std = source.std(
        dim=0,
        correction=0,
    )

    torch.testing.assert_close(
        aggregated[0],
        expected_mean,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )

    torch.testing.assert_close(
        aggregated[1],
        expected_std,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )


def _validate_single_token_case(
    device: torch.device,
) -> None:
    """Verify that a one-token sequence produces a finite zero standard deviation."""

    features = torch.tensor(
        [
            [
                1.0,
                -2.0,
                3.0,
                -4.0,
            ]
        ],
        dtype=torch.float32,
        device=device,
    )

    aggregated = mean_std_seq(
        features,
    )

    expected_mean = features[0]

    expected_std = torch.zeros_like(
        expected_mean,
    )

    torch.testing.assert_close(
        aggregated[0],
        expected_mean,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )

    torch.testing.assert_close(
        aggregated[1],
        expected_std,
        rtol=RELATIVE_TOLERANCE,
        atol=ABSOLUTE_TOLERANCE,
    )

    if not bool(
        torch.isfinite(
            aggregated,
        )
        .all()
        .item()
    ):
        raise RuntimeError("Single-token aggregation produced NaN or infinite values.")


def _run_aggregation_check(
    *,
    device: str,
) -> None:
    """Run the TACME feature aggregation smoke test."""

    resolved_device = resolve_device(
        device,
    )

    features = _create_test_features(
        resolved_device,
    )

    if tuple(
        features.shape,
    ) != (
        TEST_SEQUENCE_LENGTH,
        TEST_FEATURE_DIM,
    ):
        raise RuntimeError(f"Unexpected test feature shape: {tuple(features.shape)}.")

    aggregated = mean_std_seq(
        features,
    )

    _validate_aggregation(
        features,
        aggregated,
    )

    _validate_single_token_case(
        resolved_device,
    )

    print("TACME feature aggregation check passed.")

    print(f"Device: {resolved_device}")

    print(
        f"Input features: shape={tuple(features.shape)}, dtype={features.dtype}, device={features.device}, finite=True"
    )

    print(
        "Aggregated features: "
        f"shape={tuple(aggregated.shape)}, "
        f"dtype={aggregated.dtype}, "
        f"device={aggregated.device}, "
        "finite=True"
    )

    print("Aggregation order: mean, std")

    print("Standard deviation: population (correction=0)")

    print("Single-token sequence: passed")


def main() -> int:
    """Run the TACME feature aggregation smoke test."""

    arguments = _parse_arguments()

    try:
        _run_aggregation_check(
            device=arguments.device,
        )

    except (
        AssertionError,
        FeatureAggregationError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"TACME feature aggregation check failed: {error}",
            file=sys.stderr,
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(),
    )
