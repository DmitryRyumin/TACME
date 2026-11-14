"""
File: check_model.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Runtime smoke test for the TACME multimodal fusion model.

License: MIT License
"""

import argparse
import sys
from typing import Final

import torch
from torch import Tensor

from tacme.models import (
    MultiModalFusionModel_v2,
    TacmeModelLoadError,
    load_tacme_model,
)
from tacme.models.common import FeatureBatch

BATCH_SIZE: Final = 2
SEQUENCE_LENGTH: Final = 2

MODALITY_DIMENSIONS: Final = {
    "audio": 1024,
    "text": 768,
    "scene": 512,
    "behavior": 768,
}

EXPECTED_OUTPUT_SHAPES: Final = {
    "emotion_logits": (
        BATCH_SIZE,
        7,
    ),
    "personality_scores": (
        BATCH_SIZE,
        5,
    ),
    "ah_logits": (
        BATCH_SIZE,
        2,
    ),
}

REPEAT_RTOL: Final = 1e-6
REPEAT_ATOL: Final = 1e-7


def _parse_arguments() -> argparse.Namespace:
    """Parse runtime model-check arguments."""

    parser = argparse.ArgumentParser(
        description=("Load the TACME runtime model and verify a deterministic multimodal forward pass."),
    )

    parser.add_argument(
        "--device",
        default="auto",
        help=("PyTorch device used for the smoke test. Default: auto"),
    )

    return parser.parse_args()


def _create_feature_tensor(
    dimension: int,
    *,
    offset: float,
) -> Tensor:
    """Create one deterministic synthetic modality sequence."""

    value_count = BATCH_SIZE * SEQUENCE_LENGTH * dimension

    values = torch.linspace(
        -1.0 + offset,
        1.0 + offset,
        steps=value_count,
        dtype=torch.float32,
    )

    return values.reshape(
        BATCH_SIZE,
        SEQUENCE_LENGTH,
        dimension,
    )


def _create_test_batch() -> FeatureBatch:
    """Create deterministic multimodal input features and masks."""

    features = {
        modality: _create_feature_tensor(
            dimension,
            offset=index * 0.1,
        )
        for index, (
            modality,
            dimension,
        ) in enumerate(MODALITY_DIMENSIONS.items())
    }

    masks = {
        modality: torch.tensor(
            [
                [
                    True,
                    True,
                ],
                [
                    True,
                    False,
                ],
            ],
            dtype=torch.bool,
        )
        for modality in MODALITY_DIMENSIONS
    }

    return {
        "features": features,
        "feature_masks": masks,
    }


def _validate_model_state(
    model: MultiModalFusionModel_v2,
) -> None:
    """Validate inference-specific model state."""

    if model.training:
        raise RuntimeError("TACME runtime model is unexpectedly in training mode.")

    trainable_parameters = [name for name, parameter in model.named_parameters() if parameter.requires_grad]

    if trainable_parameters:
        msg = "TACME runtime model contains parameters with gradients enabled: " + ", ".join(
            trainable_parameters,
        )

        raise RuntimeError(msg)


def _extract_output_tensor(
    outputs: dict[str, Tensor | None],
    name: str,
) -> Tensor:
    """Return one required task output tensor."""

    value = outputs.get(
        name,
    )

    if value is None:
        msg = f"TACME model did not produce required output {name!r}."

        raise RuntimeError(msg)

    return value


def _validate_output(
    name: str,
    tensor: Tensor,
) -> None:
    """Validate shape and numerical properties of one task output."""

    expected_shape = EXPECTED_OUTPUT_SHAPES[name]

    if tuple(tensor.shape) != expected_shape:
        msg = f"Unexpected shape for {name!r}: {tuple(tensor.shape)} != {expected_shape}."

        raise RuntimeError(msg)

    if not bool(
        torch.isfinite(
            tensor,
        )
        .all()
        .item()
    ):
        msg = f"Output {name!r} contains NaN or infinite values."

        raise RuntimeError(msg)


def _validate_repeatability(
    first_outputs: dict[str, Tensor | None],
    second_outputs: dict[str, Tensor | None],
) -> None:
    """Verify stable repeated inference for identical inputs."""

    for name in EXPECTED_OUTPUT_SHAPES:
        first = _extract_output_tensor(
            first_outputs,
            name,
        )

        second = _extract_output_tensor(
            second_outputs,
            name,
        )

        try:
            torch.testing.assert_close(
                first,
                second,
                rtol=REPEAT_RTOL,
                atol=REPEAT_ATOL,
            )

        except AssertionError as error:
            msg = f"Repeated TACME inference produced different values for {name!r}."

            raise RuntimeError(msg) from error


def _run_model_check(
    *,
    device: str,
) -> None:
    """Load TACME and execute the runtime forward smoke test."""

    model = load_tacme_model(
        device=device,
    )

    _validate_model_state(
        model,
    )

    batch = _create_test_batch()

    with torch.inference_mode():
        first_outputs = model(
            batch,
        )

        second_outputs = model(
            batch,
        )

    for name in EXPECTED_OUTPUT_SHAPES:
        tensor = _extract_output_tensor(
            first_outputs,
            name,
        )

        _validate_output(
            name,
            tensor,
        )

    _validate_repeatability(
        first_outputs,
        second_outputs,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print("TACME runtime model check passed.")

    print(f"Model: {type(model).__name__}")

    print(f"Device: {model.device}")

    print(f"Parameters: {parameter_count:,}")

    for name in EXPECTED_OUTPUT_SHAPES:
        tensor = _extract_output_tensor(
            first_outputs,
            name,
        )

        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, finite=True")

    print("Repeated inference: stable")


def main() -> int:
    """Run the TACME runtime model smoke test."""

    arguments = _parse_arguments()

    try:
        _run_model_check(
            device=arguments.device,
        )

    except (
        OSError,
        RuntimeError,
        TacmeModelLoadError,
        ValueError,
    ) as error:
        print(
            f"TACME runtime model check failed: {error}",
            file=sys.stderr,
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(),
    )
