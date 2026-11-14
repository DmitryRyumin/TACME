"""
Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Runtime smoke test for the TACME Qwen3-VL behavior describer.

License: MIT License
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Final

from tacme.behavior import (
    BehaviorDescriptionError,
    QwenBehaviorDescriber,
)
from tacme.models.common import module_device

TEST_VIDEO_WIDTH: Final = 160
TEST_VIDEO_HEIGHT: Final = 120
TEST_VIDEO_FPS: Final = 8
TEST_VIDEO_FRAME_COUNT: Final = 8


def _parse_arguments() -> argparse.Namespace:
    """Parse Qwen3-VL behavior check arguments."""

    parser = argparse.ArgumentParser(
        description="Load TACME Qwen3-VL and verify behavior description generation from a deterministic test video.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device used for Qwen3-VL inference. Default: auto",
    )

    return parser.parse_args()


def _resolve_ffmpeg() -> str:
    """Return the FFmpeg executable used to create the test video."""

    executable = shutil.which(
        "ffmpeg",
    )

    if executable is None:
        raise RuntimeError("FFmpeg is required for the Qwen3-VL behavior smoke test.")

    return executable


def _create_test_video(
    path: Path,
) -> None:
    """Create a deterministic short synthetic MP4 video."""

    ffmpeg = _resolve_ffmpeg()

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        (f"testsrc=size={TEST_VIDEO_WIDTH}x{TEST_VIDEO_HEIGHT}:rate={TEST_VIDEO_FPS}"),
        "-frames:v",
        str(
            TEST_VIDEO_FRAME_COUNT,
        ),
        "-an",
        "-c:v",
        "mpeg4",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    except (
        OSError,
        subprocess.CalledProcessError,
    ) as error:
        msg = f"Unable to create the Qwen3-VL test video: {error}"
        raise RuntimeError(msg) from error

    if not path.is_file():
        raise RuntimeError("FFmpeg did not create the Qwen3-VL test video.")


def _validate_model_state(
    describer: QwenBehaviorDescriber,
) -> None:
    """Validate inference-specific Qwen3-VL model state."""

    if describer.model.training:
        raise RuntimeError("Qwen3-VL model is unexpectedly in training mode.")

    trainable_parameters = [name for name, parameter in describer.model.named_parameters() if parameter.requires_grad]

    if trainable_parameters:
        msg = "Qwen3-VL contains parameters with gradients enabled: " + ", ".join(trainable_parameters[:10])

        if (
            len(
                trainable_parameters,
            )
            > 10
        ):
            msg += ", ..."

        raise RuntimeError(msg)

    actual_device = module_device(
        describer.model,
    )

    expected_device = describer.device

    if actual_device.type != expected_device.type or (
        expected_device.index is not None and actual_device.index != expected_device.index
    ):
        msg = f"Unexpected Qwen3-VL model device: {actual_device}; expected {expected_device}."
        raise RuntimeError(msg)


def _validate_description(
    description: str,
) -> None:
    """Validate the normalized behavior description contract."""

    if not description:
        raise RuntimeError("Qwen3-VL returned an empty behavior description.")

    if "\n" in description:
        raise RuntimeError("Qwen3-VL behavior description contains a newline.")

    if "\r" in description:
        raise RuntimeError("Qwen3-VL behavior description contains a carriage return.")

    if "  " in description:
        raise RuntimeError("Qwen3-VL behavior description contains repeated spaces.")

    if description != description.strip():
        raise RuntimeError("Qwen3-VL behavior description contains leading or trailing whitespace.")

    if not description.endswith("."):
        raise RuntimeError("Qwen3-VL behavior description does not end with a period.")


def _run_behavior_check(
    *,
    device: str,
) -> None:
    """Run the TACME Qwen3-VL behavior description smoke test."""

    describer = QwenBehaviorDescriber(
        device=device,
    )

    _validate_model_state(
        describer,
    )

    with TemporaryDirectory(
        prefix="tacme-behavior-",
    ) as temporary_directory:
        video_path = (
            Path(
                temporary_directory,
            )
            / "test.mp4"
        )

        _create_test_video(
            video_path,
        )

        description = describer.describe(
            video_path,
        )

    _validate_description(
        description,
    )

    parameter = next(
        describer.model.parameters(),
        None,
    )

    dtype = parameter.dtype if parameter is not None else "unknown"

    print("TACME Qwen3-VL behavior check passed.")

    print(f"Model: {describer.model_name}")

    print(f"Device: {describer.device}")

    print(f"Dtype: {dtype}")

    print(f"Video FPS: {describer.video_fps}")

    print(f"Maximum frames: {describer.max_frames}")

    print(f"Maximum new tokens: {describer.max_new_tokens}")

    print(f"Fingerprint: {describer.fingerprint()}")

    print(f"Description: {description!r}")

    print("Normalization: passed")


def main() -> int:
    """Run the TACME Qwen3-VL behavior description smoke test."""

    arguments = _parse_arguments()

    try:
        _run_behavior_check(
            device=arguments.device,
        )

    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        BehaviorDescriptionError,
    ) as error:
        print(
            f"TACME Qwen3-VL behavior check failed: {error}",
            file=sys.stderr,
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(),
    )
