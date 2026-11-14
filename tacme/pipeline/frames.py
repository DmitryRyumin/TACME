"""
File: frames.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: FFmpeg frame probing, uniform sampling, and RGB decoding.

License: MIT License
"""

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

from PIL import Image

from tacme.pipeline.contracts import PipelineError
from tacme.pipeline.media import _find_executable


def _probe_video_frame_count(
    video_path: Path,
) -> int:
    """Return the decoded frame count of the first video stream."""

    ffprobe = _find_executable(
        "ffprobe",
    )

    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        stderr = result.stderr.strip()
        msg = f"Unable to inspect video frames with ffprobe: {stderr or 'unknown ffprobe error'}"
        raise PipelineError(msg)

    raw_count = result.stdout.strip()

    try:
        frame_count = int(
            raw_count,
        )

    except ValueError as error:
        msg = f"ffprobe returned an invalid video frame count: {raw_count!r}."
        raise PipelineError(msg) from error

    if frame_count <= 0:
        msg = f"Video contains no decodable frames: {video_path}"
        raise PipelineError(msg)

    return frame_count


def _uniform_frame_indices(
    frame_count: int,
    target_count: int,
) -> list[int]:
    """Return uniformly distributed frame indices including both endpoints."""

    if frame_count <= 0:
        msg = f"frame_count must be positive, got {frame_count}."
        raise ValueError(msg)

    if target_count <= 0:
        msg = f"target_count must be positive, got {target_count}."
        raise ValueError(msg)

    if frame_count == 1:
        return [
            0
            for _ in range(
                target_count,
            )
        ]

    if target_count == 1:
        return [
            0,
        ]

    return [
        int(index * (frame_count - 1) / (target_count - 1))
        for index in range(
            target_count,
        )
    ]


def _decode_selected_frames(
    video_path: Path,
    frame_indices: list[int],
) -> list[Image.Image]:
    """Decode selected video frames as RGB PIL images."""

    if not frame_indices:
        raise PipelineError("No frame indices were provided for scene extraction.")

    ffmpeg = _find_executable(
        "ffmpeg",
    )

    unique_indices = sorted(
        set(
            frame_indices,
        )
    )

    select_expression = "+".join(f"eq(n\\,{frame_index})" for frame_index in unique_indices)

    with TemporaryDirectory(
        prefix="tacme-scene-",
    ) as temporary_directory:
        output_pattern = (
            Path(
                temporary_directory,
            )
            / "frame-%06d.png"
        )

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-vf",
            f"select={select_expression}",
            "-fps_mode",
            "vfr",
            str(output_pattern),
        ]

        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip()
            msg = f"Unable to decode scene frames with FFmpeg: {stderr or 'unknown FFmpeg error'}"
            raise PipelineError(msg)

        frame_paths = sorted(
            Path(
                temporary_directory,
            ).glob(
                "frame-*.png",
            )
        )

        if len(
            frame_paths,
        ) != len(
            unique_indices,
        ):
            msg = f"FFmpeg decoded {len(frame_paths)} selected frames; expected {len(unique_indices)}."
            raise PipelineError(msg)

        decoded_frames: dict[int, Image.Image] = {}

        for frame_index, frame_path in zip(
            unique_indices,
            frame_paths,
            strict=True,
        ):
            try:
                with Image.open(
                    frame_path,
                ) as image:
                    decoded_frames[frame_index] = image.convert(
                        "RGB",
                    ).copy()

            except OSError as error:
                msg = f"Unable to read decoded scene frame {frame_path}: {error}"
                raise PipelineError(msg) from error

    try:
        return [decoded_frames[frame_index] for frame_index in frame_indices]

    except KeyError as error:
        msg = f"Decoded scene frame {error.args[0]} is missing."
        raise PipelineError(msg) from error


def _sample_scene_frames_with_indices(
    video_path: Path,
    frame_count: int,
) -> tuple[
    list[Image.Image],
    list[int],
    int,
]:
    """Uniformly sample scene frames and preserve their original indices."""

    total_frames = _probe_video_frame_count(
        video_path,
    )

    frame_indices = _uniform_frame_indices(
        total_frames,
        frame_count,
    )

    frames = _decode_selected_frames(
        video_path,
        frame_indices,
    )

    return (
        frames,
        frame_indices,
        total_frames,
    )


def _sample_scene_frames(
    video_path: Path,
    frame_count: int,
) -> list[Image.Image]:
    """Uniformly sample scene frames from one video."""

    frames, _, _ = _sample_scene_frames_with_indices(
        video_path,
        frame_count,
    )

    return frames
