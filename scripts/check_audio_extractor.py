"""
File: check_audio_extractor.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Runtime smoke test for the TACME EmoWav2Vec2 audio feature extractor.

License: MIT License
"""

import argparse
import math
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
from typing import Final
import wave

import torch
from torch import Tensor

from tacme.extractors import (
    EXPECTED_AUDIO_DIM,
    AudioExtractionError,
    EmoWav2Vec2Extractor,
)

TEST_DURATION_SECONDS: Final = 2.0
TEST_FREQUENCY_HZ: Final = 440.0
TEST_AMPLITUDE: Final = 0.25
TEST_SAMPLE_WIDTH_BYTES: Final = 2


def _parse_arguments() -> argparse.Namespace:
    """Parse audio extractor check arguments."""

    parser = argparse.ArgumentParser(
        description=("Load the TACME EmoWav2Vec2 extractor and verify feature extraction on synthetic audio."),
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device used for inference. Default: auto",
    )

    return parser.parse_args()


def _create_test_audio(
    path: Path,
    *,
    sample_rate: int,
) -> None:
    """Create a deterministic mono PCM WAV file."""

    sample_count = int(TEST_DURATION_SECONDS * sample_rate)

    pcm_samples = bytearray()

    for index in range(
        sample_count,
    ):
        time_seconds = index / sample_rate

        value = TEST_AMPLITUDE * math.sin(2.0 * math.pi * TEST_FREQUENCY_HZ * time_seconds)

        pcm_value = int(value * 32767)

        pcm_samples.extend(
            struct.pack(
                "<h",
                pcm_value,
            )
        )

    with wave.open(
        str(path),
        "wb",
    ) as audio_file:
        audio_file.setnchannels(
            1,
        )

        audio_file.setsampwidth(
            TEST_SAMPLE_WIDTH_BYTES,
        )

        audio_file.setframerate(
            sample_rate,
        )

        audio_file.writeframes(
            pcm_samples,
        )


def _validate_features(
    features: Tensor,
    *,
    expected_device: torch.device,
) -> None:
    """Validate extracted EmoWav2Vec2 features."""

    if features.ndim != 2:
        msg = f"Unexpected audio feature rank: {features.ndim}; expected 2."
        raise RuntimeError(msg)

    if features.shape[0] <= 0:
        raise RuntimeError("Audio extractor returned an empty sequence.")

    if features.shape[1] != EXPECTED_AUDIO_DIM:
        msg = f"Unexpected audio feature dimension: {features.shape[1]}; expected {EXPECTED_AUDIO_DIM}."
        raise RuntimeError(msg)

    if features.dtype != torch.float32:
        msg = f"Unexpected audio feature dtype: {features.dtype}; expected torch.float32."
        raise RuntimeError(msg)

    if features.device.type != expected_device.type or (
        expected_device.index is not None and features.device.index != expected_device.index
    ):
        msg = f"Unexpected audio feature device: {features.device}; expected {expected_device}."
        raise RuntimeError(msg)

    if not bool(
        torch.isfinite(
            features,
        )
        .all()
        .item()
    ):
        raise RuntimeError("Audio features contain NaN or infinite values.")


def _validate_model_state(
    extractor: EmoWav2Vec2Extractor,
) -> None:
    """Validate inference-specific extractor state."""

    if extractor.model.training:
        raise RuntimeError("EmoWav2Vec2 model is unexpectedly in training mode.")

    trainable_parameters = [name for name, parameter in extractor.model.named_parameters() if parameter.requires_grad]

    if trainable_parameters:
        msg = "EmoWav2Vec2 contains parameters with gradients enabled: " + ", ".join(
            trainable_parameters,
        )
        raise RuntimeError(msg)


def _run_audio_check(
    *,
    device: str,
) -> None:
    """Run the TACME audio extractor smoke test."""

    extractor = EmoWav2Vec2Extractor(
        device=device,
    )

    _validate_model_state(
        extractor,
    )

    with TemporaryDirectory(
        prefix="tacme-audio-",
    ) as temporary_directory:
        audio_path = (
            Path(
                temporary_directory,
            )
            / "test.wav"
        )

        _create_test_audio(
            audio_path,
            sample_rate=extractor.sample_rate,
        )

        features = extractor.extract(
            audio_path,
        )

    _validate_features(
        features,
        expected_device=extractor.device,
    )

    print("TACME audio extractor check passed.")

    print(f"Model: {extractor.model_name}")

    print(f"Device: {extractor.device}")

    print(f"Sample rate: {extractor.sample_rate} Hz")

    print(f"Features: shape={tuple(features.shape)}, dtype={features.dtype}, device={features.device}, finite=True")


def main() -> int:
    """Run the TACME audio extractor smoke test."""

    arguments = _parse_arguments()

    try:
        _run_audio_check(
            device=arguments.device,
        )

    except (
        AudioExtractionError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(
            f"TACME audio extractor check failed: {error}",
            file=sys.stderr,
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(),
    )
