"""
File: check_pipeline.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: End-to-end runtime smoke test for the TACME multimodal feature pipeline.

License: MIT License
"""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Final

import torch
from torch import Tensor

from tacme.models import load_tacme_model
from tacme.models.common import FeatureBatch
from tacme.pipeline import (
    AGGREGATED_SEQUENCE_LENGTH,
    MODALITY_DIMENSIONS,
    TacmeFeaturePipeline,
)

TEST_VIDEO_WIDTH: Final = 320
TEST_VIDEO_HEIGHT: Final = 240
TEST_VIDEO_FPS: Final = 8
TEST_VIDEO_DURATION_SECONDS: Final = 1.0
TEST_AUDIO_SAMPLE_RATE: Final = 16000
TEST_AUDIO_FREQUENCY_HZ: Final = 1000

EXPECTED_OUTPUT_SHAPES: Final = {
    "emotion_logits": (
        1,
        7,
    ),
    "personality_scores": (
        1,
        5,
    ),
    "ah_logits": (
        1,
        2,
    ),
}


def _parse_arguments() -> argparse.Namespace:
    """Parse pipeline-check arguments."""

    parser = argparse.ArgumentParser(
        description="Run the complete TACME video-to-model inference pipeline.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device used for the smoke test. Default: auto",
    )

    return parser.parse_args()


def _create_test_video(
    path: Path,
) -> None:
    """Create one deterministic synthetic video with an audio track."""

    ffmpeg = shutil.which(
        "ffmpeg",
    )

    if ffmpeg is None:
        raise RuntimeError("FFmpeg was not found in PATH.")

    video_source = (
        f"testsrc2="
        f"size={TEST_VIDEO_WIDTH}x{TEST_VIDEO_HEIGHT}:"
        f"rate={TEST_VIDEO_FPS}:"
        f"duration={TEST_VIDEO_DURATION_SECONDS}"
    )

    audio_source = (
        f"sine="
        f"frequency={TEST_AUDIO_FREQUENCY_HZ}:"
        f"sample_rate={TEST_AUDIO_SAMPLE_RATE}:"
        f"duration={TEST_VIDEO_DURATION_SECONDS}"
    )

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        video_source,
        "-f",
        "lavfi",
        "-i",
        audio_source,
        "-shortest",
        "-c:v",
        "mpeg4",
        "-q:v",
        "5",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        str(path),
    ]

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        msg = f"Unable to create synthetic TACME test video: {stderr or 'unknown FFmpeg error'}"
        raise RuntimeError(msg)

    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError("Synthetic TACME test video was not created.")


def _validate_feature_batch(
    batch: FeatureBatch,
    *,
    expected_device: torch.device,
) -> None:
    """Validate the complete TACME feature batch."""

    features = batch["features"]

    expected_modalities = set(
        MODALITY_DIMENSIONS,
    )

    actual_modalities = set(
        features,
    )

    if actual_modalities != expected_modalities:
        msg = f"Unexpected feature modalities: {sorted(actual_modalities)}; expected {sorted(expected_modalities)}."
        raise RuntimeError(msg)

    masks = batch.get(
        "feature_masks",
    )

    if masks is None:
        raise RuntimeError("Pipeline did not return feature masks.")

    if (
        set(
            masks,
        )
        != expected_modalities
    ):
        msg = f"Unexpected feature-mask modalities: {sorted(masks)}."
        raise RuntimeError(msg)

    for modality, expected_dim in MODALITY_DIMENSIONS.items():
        feature = features[modality]

        expected_shape = (
            1,
            AGGREGATED_SEQUENCE_LENGTH,
            expected_dim,
        )

        if (
            tuple(
                feature.shape,
            )
            != expected_shape
        ):
            msg = f"Unexpected {modality!r} feature shape: {tuple(feature.shape)}; expected {expected_shape}."
            raise RuntimeError(msg)

        if feature.dtype != torch.float32:
            msg = f"Unexpected {modality!r} dtype: {feature.dtype}; expected torch.float32."
            raise RuntimeError(msg)

        if feature.device.type != expected_device.type or (
            expected_device.index is not None and feature.device.index != expected_device.index
        ):
            msg = f"Unexpected {modality!r} device: {feature.device}; expected {expected_device}."
            raise RuntimeError(msg)

        if not bool(
            torch.isfinite(
                feature,
            )
            .all()
            .item()
        ):
            msg = f"{modality!r} features contain NaN or infinite values."
            raise RuntimeError(msg)

        mask = masks[modality]

        expected_mask_shape = (
            1,
            AGGREGATED_SEQUENCE_LENGTH,
        )

        if (
            tuple(
                mask.shape,
            )
            != expected_mask_shape
        ):
            msg = f"Unexpected {modality!r} mask shape: {tuple(mask.shape)}; expected {expected_mask_shape}."
            raise RuntimeError(msg)

        if mask.dtype != torch.bool:
            msg = f"Unexpected {modality!r} mask dtype: {mask.dtype}; expected torch.bool."
            raise RuntimeError(msg)

        if mask.device != feature.device:
            msg = f"{modality!r} mask is on {mask.device}, but its features are on {feature.device}."
            raise RuntimeError(msg)

        if not bool(mask.all().item()):
            msg = f"{modality!r} mask unexpectedly contains invalid tokens."
            raise RuntimeError(msg)


def _validate_model_outputs(
    outputs: dict[str, Tensor | None],
) -> None:
    """Validate TACME task outputs produced from pipeline features."""

    for name, expected_shape in EXPECTED_OUTPUT_SHAPES.items():
        tensor = outputs.get(
            name,
        )

        if tensor is None:
            msg = f"TACME did not produce required output {name!r}."
            raise RuntimeError(msg)

        if (
            tuple(
                tensor.shape,
            )
            != expected_shape
        ):
            msg = f"Unexpected {name!r} shape: {tuple(tensor.shape)}; expected {expected_shape}."
            raise RuntimeError(msg)

        if not bool(
            torch.isfinite(
                tensor,
            )
            .all()
            .item()
        ):
            msg = f"{name!r} contains NaN or infinite values."
            raise RuntimeError(msg)


def _run_pipeline_check(
    *,
    device: str,
) -> None:
    """Run complete TACME media preparation and model inference."""

    pipeline = TacmeFeaturePipeline(
        device=device,
    )

    with TemporaryDirectory(
        prefix="tacme-pipeline-",
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

        batch = pipeline.process(
            video_path,
        )

    _validate_feature_batch(
        batch,
        expected_device=pipeline.device,
    )

    model = load_tacme_model(
        device=str(
            pipeline.device,
        ),
    )

    with torch.inference_mode():
        outputs = model(
            batch,
        )

    _validate_model_outputs(
        outputs,
    )

    print("TACME end-to-end pipeline check passed.")
    print(f"Device: {pipeline.device}")
    print(f"Scene frames: {pipeline.scene_frame_count}")
    print("Aggregation: mean_std_seq")
    print(f"Aggregated sequence length: {AGGREGATED_SEQUENCE_LENGTH}")

    masks = batch.get(
        "feature_masks",
    )

    if masks is None:
        raise RuntimeError("Pipeline did not return feature masks.")

    for modality in MODALITY_DIMENSIONS:
        feature = batch["features"][modality]
        mask = masks[modality]

        print(
            f"{modality}: "
            f"shape={tuple(feature.shape)}, "
            f"dtype={feature.dtype}, "
            f"device={feature.device}, "
            f"mask={tuple(mask.shape)}, "
            "finite=True"
        )

    for name in EXPECTED_OUTPUT_SHAPES:
        tensor = outputs[name]

        if tensor is None:
            raise RuntimeError(f"Missing validated output {name!r}.")

        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, finite=True")


def main() -> int:
    """Run the TACME pipeline smoke test."""

    arguments = _parse_arguments()

    try:
        _run_pipeline_check(
            device=arguments.device,
        )

    except (
        ImportError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(
            f"TACME end-to-end pipeline check failed: {error}",
            file=sys.stderr,
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(),
    )
