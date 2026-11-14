"""
Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Input-video validation and Qwen3-VL visual preprocessing helpers.

License: MIT License
"""

from collections.abc import Callable
import importlib
import os
from pathlib import Path
from typing import cast

from tacme.behavior.contracts import QWEN3_VL_VIDEO_READER, BehaviorDescriptionError


def _validate_video_path(
    video_path: Path,
) -> Path:
    """Resolve and validate one input video path."""

    path = video_path.expanduser().resolve()

    if not path.is_file():
        msg = f"Video file does not exist: {path}"
        raise BehaviorDescriptionError(msg)

    return path


def _load_process_vision_info() -> Callable[..., object]:
    """Load qwen-vl-utils with the TACME video decoding backend."""

    os.environ["FORCE_QWENVL_VIDEO_READER"] = QWEN3_VL_VIDEO_READER

    try:
        module = importlib.import_module(
            "qwen_vl_utils",
        )

    except ImportError as error:
        msg = f"Unable to import qwen-vl-utils: {error}"
        raise BehaviorDescriptionError(msg) from error

    function = getattr(
        module,
        "process_vision_info",
        None,
    )

    if not callable(
        function,
    ):
        raise BehaviorDescriptionError("qwen-vl-utils does not expose process_vision_info.")

    return cast(
        Callable[..., object],
        function,
    )


def _split_video_metadata(
    videos: object | None,
) -> tuple[
    object | None,
    list[object] | None,
]:
    """Split Qwen3-VL video tensors from their video metadata."""

    if videos is None:
        return (
            None,
            None,
        )

    if not isinstance(
        videos,
        (
            list,
            tuple,
        ),
    ):
        msg = f"Unexpected Qwen3-VL video input type: {type(videos).__name__}."
        raise BehaviorDescriptionError(msg)

    video_inputs: list[object] = []
    video_metadata: list[object] = []

    for item in videos:
        if (
            not isinstance(
                item,
                (
                    list,
                    tuple,
                ),
            )
            or len(item) != 2
        ):
            raise BehaviorDescriptionError("Qwen3-VL video preprocessing did not return video-metadata pairs.")

        video_inputs.append(
            item[0],
        )

        video_metadata.append(
            item[1],
        )

    return (
        video_inputs,
        video_metadata,
    )
