"""
Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Runtime smoke test for the TACME Whisper speech transcriber.

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

from tacme.models.common import module_device
from tacme.transcription import (
    WhisperTranscriber,
    WhisperTranscriptionError,
)

TEST_SAMPLE_RATE: Final = 16_000
TEST_DURATION_SECONDS: Final = 1.0
TEST_FREQUENCY_HZ: Final = 440.0
TEST_AMPLITUDE: Final = 0.1
TEST_SAMPLE_WIDTH_BYTES: Final = 2


def _parse_arguments() -> argparse.Namespace:
    """Parse Whisper transcriber check arguments."""

    parser = argparse.ArgumentParser(
        description="Load the TACME Whisper transcriber and verify inference on deterministic synthetic audio.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device used for Whisper inference. Default: auto",
    )

    return parser.parse_args()


def _create_test_audio(
    path: Path,
) -> None:
    """Create a deterministic mono PCM WAV file."""

    sample_count = int(
        TEST_DURATION_SECONDS * TEST_SAMPLE_RATE,
    )

    pcm_samples = bytearray()

    for index in range(
        sample_count,
    ):
        time_seconds = index / TEST_SAMPLE_RATE

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
            TEST_SAMPLE_RATE,
        )

        audio_file.writeframes(
            pcm_samples,
        )


def _validate_model_state(
    transcriber: WhisperTranscriber,
) -> None:
    """Validate inference-specific Whisper state."""

    if transcriber.model.training:
        raise RuntimeError("Whisper model is unexpectedly in training mode.")

    trainable_parameters = [name for name, parameter in transcriber.model.named_parameters() if parameter.requires_grad]

    if trainable_parameters:
        msg = "Whisper contains parameters with gradients enabled: " + ", ".join(trainable_parameters)
        raise RuntimeError(msg)

    actual_device = module_device(
        transcriber.model,
    )

    if actual_device.type != transcriber.device.type or (
        transcriber.device.index is not None and actual_device.index != transcriber.device.index
    ):
        msg = f"Unexpected Whisper model device: {actual_device}; expected {transcriber.device}."
        raise RuntimeError(msg)


def _validate_transcription(
    transcription: str,
) -> None:
    """Validate normalized Whisper transcription output."""

    if "\n" in transcription:
        raise RuntimeError("Whisper transcription contains a newline.")

    if "\r" in transcription:
        raise RuntimeError("Whisper transcription contains a carriage return.")

    if "  " in transcription:
        raise RuntimeError("Whisper transcription contains repeated spaces.")

    if transcription != transcription.strip():
        raise RuntimeError("Whisper transcription contains leading or trailing whitespace.")

    if transcription and not transcription.endswith("."):
        raise RuntimeError("Non-empty Whisper transcription does not end with a period.")


def _run_transcription_check(
    *,
    device: str,
) -> None:
    """Run the TACME Whisper transcription smoke test."""

    transcriber = WhisperTranscriber(
        device=device,
    )

    _validate_model_state(
        transcriber,
    )

    with TemporaryDirectory(
        prefix="tacme-whisper-",
    ) as temporary_directory:
        audio_path = (
            Path(
                temporary_directory,
            )
            / "test.wav"
        )

        _create_test_audio(
            audio_path,
        )

        transcription = transcriber.transcribe(
            audio_path,
        )

    _validate_transcription(
        transcription,
    )

    print("TACME Whisper transcription check passed.")

    print(f"Model: {transcriber.model_name}")

    print(f"Device: {transcriber.device}")

    print(f"Sample rate: {TEST_SAMPLE_RATE} Hz")

    print(f"Fingerprint: {transcriber.fingerprint()}")

    print(f"Transcription: {transcription!r}")

    print("Normalization: passed")


def main() -> int:
    """Run the TACME Whisper transcription smoke test."""

    arguments = _parse_arguments()

    try:
        _run_transcription_check(
            device=arguments.device,
        )

    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        WhisperTranscriptionError,
    ) as error:
        print(
            f"TACME Whisper transcription check failed: {error}",
            file=sys.stderr,
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(),
    )
