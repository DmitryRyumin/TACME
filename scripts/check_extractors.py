"""
Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Runtime smoke test for the TACME multimodal feature extractors.

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

from PIL import Image
import torch
from torch import Tensor, nn

from tacme.extractors import (
    EXPECTED_AUDIO_DIM,
    EXPECTED_TEXT_DIM,
    EXPECTED_VISION_DIM,
    AudioExtractionError,
    CLIPSceneExtractor,
    EmoRoBERTaExtractor,
    EmoWav2Vec2Extractor,
    TextExtractionError,
    VisionExtractionError,
)

TEST_FRAME_COUNT: Final = 3
TEST_FRAME_SIZE: Final = (320, 240)

TEST_DURATION_SECONDS: Final = 2.0
TEST_FREQUENCY_HZ: Final = 440.0
TEST_AMPLITUDE: Final = 0.25
TEST_SAMPLE_WIDTH_BYTES: Final = 2

TRANSCRIPT_TEXT: Final = "The participant describes a recent experience."
BEHAVIOR_TEXT: Final = "The participant looks toward the camera and moves both hands while speaking."


def _parse_arguments() -> argparse.Namespace:
    """Parse multimodal extractor check arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Load all TACME feature extractors and verify vision, audio, transcript, and behavior feature extraction."
        ),
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device used for inference. Default: auto",
    )

    return parser.parse_args()


def _create_test_frames() -> list[Image.Image]:
    """Create deterministic RGB frames for CLIP extraction."""

    frames: list[Image.Image] = []

    for index in range(TEST_FRAME_COUNT):
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
            TEST_FRAME_SIZE,
            (
                value,
                255 - value,
                value // 2,
            ),
        )

        frames.append(frame)

    return frames


def _create_test_audio(
    path: Path,
    *,
    sample_rate: int,
) -> None:
    """Create a deterministic mono PCM WAV file."""

    sample_count = int(
        TEST_DURATION_SECONDS * sample_rate,
    )

    pcm_samples = bytearray()

    for index in range(sample_count):
        time_seconds = index / sample_rate

        value = TEST_AMPLITUDE * math.sin(
            2.0 * math.pi * TEST_FREQUENCY_HZ * time_seconds,
        )

        pcm_value = int(
            value * 32767,
        )

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
    expected_dim: int,
    expected_device: torch.device,
    stream_name: str,
) -> None:
    """Validate one TACME feature sequence."""

    if features.ndim != 2:
        msg = f"{stream_name} feature rank is {features.ndim}; expected 2."
        raise RuntimeError(msg)

    if features.shape[0] <= 0:
        raise RuntimeError(
            f"{stream_name} feature sequence is empty.",
        )

    if features.shape[1] != expected_dim:
        msg = f"{stream_name} feature dimension is {features.shape[1]}; expected {expected_dim}."
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
        raise RuntimeError(
            f"{stream_name} features contain NaN or infinite values.",
        )


def _validate_model_state(
    model: nn.Module,
    *,
    model_name: str,
) -> None:
    """Validate inference-specific model state."""

    if model.training:
        raise RuntimeError(
            f"{model_name} is unexpectedly in training mode.",
        )

    trainable_parameters = [name for name, parameter in model.named_parameters() if parameter.requires_grad]

    if trainable_parameters:
        msg = f"{model_name} contains parameters with gradients enabled: " + ", ".join(trainable_parameters)
        raise RuntimeError(msg)


def _run_extractors_check(
    *,
    device: str,
) -> None:
    """Run the TACME multimodal extractor smoke test."""

    vision_extractor = CLIPSceneExtractor(
        device=device,
    )

    audio_extractor = EmoWav2Vec2Extractor(
        device=device,
    )

    text_extractor = EmoRoBERTaExtractor(
        device=device,
    )

    if vision_extractor.device != audio_extractor.device or vision_extractor.device != text_extractor.device:
        msg = (
            "TACME extractors resolved different devices: "
            f"vision={vision_extractor.device}, "
            f"audio={audio_extractor.device}, "
            f"text={text_extractor.device}."
        )
        raise RuntimeError(msg)

    _validate_model_state(
        vision_extractor.model,
        model_name="CLIP vision model",
    )

    _validate_model_state(
        audio_extractor.model,
        model_name="EmoWav2Vec2 model",
    )

    _validate_model_state(
        text_extractor.model,
        model_name="EmoRoBERTa model",
    )

    frames = _create_test_frames()

    try:
        vision_features = vision_extractor.extract(
            frames,
        )

    finally:
        for frame in frames:
            frame.close()

    with TemporaryDirectory(
        prefix="tacme-extractors-",
    ) as temporary_directory:
        audio_path = (
            Path(
                temporary_directory,
            )
            / "test.wav"
        )

        _create_test_audio(
            audio_path,
            sample_rate=audio_extractor.sample_rate,
        )

        audio_features = audio_extractor.extract(
            audio_path,
        )

    transcript_features = text_extractor.extract(
        TRANSCRIPT_TEXT,
    )

    behavior_features = text_extractor.extract(
        BEHAVIOR_TEXT,
    )

    _validate_features(
        vision_features,
        expected_dim=EXPECTED_VISION_DIM,
        expected_device=vision_extractor.device,
        stream_name="Vision",
    )

    _validate_features(
        audio_features,
        expected_dim=EXPECTED_AUDIO_DIM,
        expected_device=audio_extractor.device,
        stream_name="Audio",
    )

    _validate_features(
        transcript_features,
        expected_dim=EXPECTED_TEXT_DIM,
        expected_device=text_extractor.device,
        stream_name="Transcript",
    )

    _validate_features(
        behavior_features,
        expected_dim=EXPECTED_TEXT_DIM,
        expected_device=text_extractor.device,
        stream_name="Behavior",
    )

    print("TACME multimodal extractor check passed.")

    print(f"Device: {vision_extractor.device}")

    print(f"Vision model: {vision_extractor.model_name}")
    print(f"Audio model: {audio_extractor.model_name}")
    print(f"Text model: {text_extractor.model_name}")

    print("Shared text extractor instance: yes")

    print(
        "Vision features: "
        f"shape={tuple(vision_features.shape)}, "
        f"dtype={vision_features.dtype}, "
        f"device={vision_features.device}, "
        "finite=True"
    )

    print(
        "Audio features: "
        f"shape={tuple(audio_features.shape)}, "
        f"dtype={audio_features.dtype}, "
        f"device={audio_features.device}, "
        "finite=True"
    )

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
    """Run the TACME multimodal extractor smoke test."""

    arguments = _parse_arguments()

    try:
        _run_extractors_check(
            device=arguments.device,
        )

    except (
        AudioExtractionError,
        OSError,
        RuntimeError,
        TextExtractionError,
        ValueError,
        VisionExtractionError,
    ) as error:
        print(
            f"TACME multimodal extractor check failed: {error}",
            file=sys.stderr,
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(),
    )
