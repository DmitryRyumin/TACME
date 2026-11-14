"""
File: check_text_extractor.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Runtime smoke test for the shared TACME EmoRoBERTa text feature extractor.

License: MIT License
"""

import argparse
import sys

import torch
from torch import Tensor

from tacme.extractors import (
    EXPECTED_TEXT_DIM,
    EmoRoBERTaExtractor,
    TextExtractionError,
)

TRANSCRIPT_TEXT = "The participant describes a recent experience."
BEHAVIOR_TEXT = "The participant looks toward the camera and moves both hands while speaking."


def _parse_arguments() -> argparse.Namespace:
    """Parse text extractor check arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Load one shared TACME EmoRoBERTa extractor and verify transcript and behavior feature extraction."
        ),
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device used for inference. Default: auto",
    )

    return parser.parse_args()


def _validate_features(
    features: Tensor,
    *,
    expected_device: torch.device,
    stream_name: str,
) -> None:
    """Validate one extracted EmoRoBERTa feature sequence."""

    if features.ndim != 2:
        msg = f"{stream_name} feature rank is {features.ndim}; expected 2."
        raise RuntimeError(msg)

    if features.shape[0] <= 0:
        msg = f"{stream_name} feature sequence is empty."
        raise RuntimeError(msg)

    if features.shape[1] != EXPECTED_TEXT_DIM:
        msg = f"{stream_name} feature dimension is {features.shape[1]}; expected {EXPECTED_TEXT_DIM}."
        raise RuntimeError(msg)

    if features.dtype != torch.float32:
        msg = f"{stream_name} feature dtype is {features.dtype}; expected torch.float32."
        raise RuntimeError(msg)

    if features.device.type != expected_device.type or (
        expected_device.index is not None and features.device.index != expected_device.index
    ):
        msg = f"{stream_name} feature device is {features.device}; expected {expected_device}."
        raise RuntimeError(msg)

    if not bool(
        torch.isfinite(
            features,
        )
        .all()
        .item()
    ):
        msg = f"{stream_name} features contain NaN or infinite values."
        raise RuntimeError(msg)


def _validate_model_state(
    extractor: EmoRoBERTaExtractor,
) -> None:
    """Validate inference-specific extractor state."""

    if extractor.model.training:
        raise RuntimeError(
            "EmoRoBERTa model is unexpectedly in training mode.",
        )

    pooler = getattr(
        extractor.model,
        "pooler",
        None,
    )

    if pooler is not None:
        raise RuntimeError(
            "EmoRoBERTa pooler is unexpectedly enabled.",
        )

    trainable_parameters = [name for name, parameter in extractor.model.named_parameters() if parameter.requires_grad]

    if trainable_parameters:
        msg = "EmoRoBERTa contains parameters with gradients enabled: " + ", ".join(trainable_parameters)
        raise RuntimeError(msg)


def _run_text_check(
    *,
    device: str,
) -> None:
    """Run the shared TACME text extractor smoke test."""

    extractor = EmoRoBERTaExtractor(
        device=device,
    )

    _validate_model_state(
        extractor,
    )

    transcript_features = extractor.extract(
        TRANSCRIPT_TEXT,
    )

    behavior_features = extractor.extract(
        BEHAVIOR_TEXT,
    )

    _validate_features(
        transcript_features,
        expected_device=extractor.device,
        stream_name="Transcript",
    )

    _validate_features(
        behavior_features,
        expected_device=extractor.device,
        stream_name="Behavior",
    )

    print("TACME text extractor check passed.")

    print(f"Model: {extractor.model_name}")

    print(f"Device: {extractor.device}")

    print(f"Max length: {extractor.max_length}")

    print("Shared extractor instance: yes")

    print(
        "Transcript features: "
        f"shape={tuple(transcript_features.shape)}, "
        f"dtype={transcript_features.dtype}, "
        f"device={transcript_features.device}, "
        "finite=True"
    )

    print(
        "Behavior features: "
        f"shape={tuple(behavior_features.shape)}, "
        f"dtype={behavior_features.dtype}, "
        f"device={behavior_features.device}, "
        "finite=True"
    )


def main() -> int:
    """Run the TACME text extractor smoke test."""

    arguments = _parse_arguments()

    try:
        _run_text_check(
            device=arguments.device,
        )

    except (
        OSError,
        RuntimeError,
        TextExtractionError,
        ValueError,
    ) as error:
        print(
            f"TACME text extractor check failed: {error}",
            file=sys.stderr,
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(),
    )
