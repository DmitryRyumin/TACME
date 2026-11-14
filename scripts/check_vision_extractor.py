"""
File: check_vision_extractor.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Runtime smoke test for the TACME CLIP scene feature extractor.

License: MIT License
"""

import argparse
import sys
from typing import Final

from PIL import Image
import torch
from torch import Tensor

from tacme.extractors import (
    EXPECTED_VISION_DIM,
    CLIPSceneExtractor,
    VisionExtractionError,
)

TEST_FRAME_COUNT: Final = 3
TEST_FRAME_WIDTH: Final = 320
TEST_FRAME_HEIGHT: Final = 240


def _parse_arguments() -> argparse.Namespace:
    """Parse vision extractor check arguments."""

    parser = argparse.ArgumentParser(
        description=("Load the TACME CLIP scene extractor and verify image-feature extraction."),
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device used for inference. Default: auto",
    )

    return parser.parse_args()


def _create_test_frames() -> list[Image.Image]:
    """Create deterministic RGB frames for the smoke test."""

    frames: list[Image.Image] = []

    for index in range(
        TEST_FRAME_COUNT,
    ):
        value = int(
            255
            * index
            / max(
                TEST_FRAME_COUNT - 1,
                1,
            )
        )

        frame = Image.new(
            "RGB",
            (
                TEST_FRAME_WIDTH,
                TEST_FRAME_HEIGHT,
            ),
            (
                value,
                255 - value,
                value // 2,
            ),
        )

        frames.append(
            frame,
        )

    return frames


def _validate_features(
    features: Tensor,
    *,
    expected_device: torch.device,
) -> None:
    """Validate extracted CLIP scene features."""

    if features.ndim != 2:
        msg = f"Unexpected visual feature rank: {features.ndim}; expected 2."
        raise RuntimeError(msg)

    if features.shape[0] != TEST_FRAME_COUNT:
        msg = f"Unexpected visual sequence length: {features.shape[0]}; expected {TEST_FRAME_COUNT}."
        raise RuntimeError(msg)

    if features.shape[1] != EXPECTED_VISION_DIM:
        msg = f"Unexpected visual feature dimension: {features.shape[1]}; expected {EXPECTED_VISION_DIM}."
        raise RuntimeError(msg)

    if features.dtype != torch.float32:
        msg = f"Unexpected visual feature dtype: {features.dtype}; expected torch.float32."
        raise RuntimeError(msg)

    if features.device.type != expected_device.type or (
        expected_device.index is not None and features.device.index != expected_device.index
    ):
        msg = f"Unexpected visual feature device: {features.device}; expected {expected_device}."
        raise RuntimeError(msg)

    if not bool(
        torch.isfinite(
            features,
        )
        .all()
        .item()
    ):
        raise RuntimeError("Visual features contain NaN or infinite values.")


def _validate_model_state(
    extractor: CLIPSceneExtractor,
) -> None:
    """Validate inference-specific extractor state."""

    if extractor.model.training:
        raise RuntimeError("CLIP model is unexpectedly in training mode.")

    trainable_parameters = [name for name, parameter in extractor.model.named_parameters() if parameter.requires_grad]

    if trainable_parameters:
        msg = "CLIP contains parameters with gradients enabled: " + ", ".join(
            trainable_parameters,
        )
        raise RuntimeError(msg)


def _run_vision_check(
    *,
    device: str,
) -> None:
    """Run the TACME CLIP scene extractor smoke test."""

    extractor = CLIPSceneExtractor(
        device=device,
    )

    _validate_model_state(
        extractor,
    )

    frames = _create_test_frames()

    features = extractor.extract(
        frames,
    )

    _validate_features(
        features,
        expected_device=extractor.device,
    )

    print("TACME vision extractor check passed.")

    print(f"Model: {extractor.model_name}")

    print(f"Device: {extractor.device}")

    print(f"Frames: {len(frames)}")

    print(f"Features: shape={tuple(features.shape)}, dtype={features.dtype}, device={features.device}, finite=True")


def main() -> int:
    """Run the TACME CLIP scene extractor smoke test."""

    arguments = _parse_arguments()

    try:
        _run_vision_check(
            device=arguments.device,
        )

    except (
        OSError,
        RuntimeError,
        VisionExtractionError,
        ValueError,
    ) as error:
        print(
            f"TACME vision extractor check failed: {error}",
            file=sys.stderr,
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(),
    )
