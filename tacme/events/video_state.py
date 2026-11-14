"""
File: video_state.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Shared video-input state helpers for TACME application events.

License: MIT License
"""

from pathlib import Path
from typing import Any, Final

import gradio as gr

from tacme.localization import (
    get_language_index,
    get_localized_text,
)

RECORDED_VIDEO_FILENAME: Final = "recording.mp4"


def get_video_display_filename(
    video_path: str | None,
) -> str | None:
    """Return a safe display filename for a video path."""

    if not video_path:
        return None

    filename = Path(
        video_path,
    ).name.strip()

    return filename or None


def resolve_video_display_filename(
    video_path: str | None,
    video_filename: str | None,
) -> str | None:
    """Resolve the display filename for the current video sample."""

    if not video_path:
        return None

    if video_filename:
        normalized_filename = Path(
            video_filename,
        ).name.strip()

        if normalized_filename:
            return normalized_filename

    return get_video_display_filename(
        video_path,
    )


def create_video_component_label(
    language_index: int,
    video_filename: str | None,
) -> str:
    """Create the video-component label for the current sample."""

    if video_filename:
        return video_filename

    return get_localized_text(
        "Labels_VIDEO_INPUT",
        language_index,
    ).strip()


def handle_video_filename_change(
    video_path: str | None,
    language: str,
    *,
    video_filename: str | None = None,
) -> tuple[Any, str | None]:
    """Update the video label and stored display filename."""

    language_index = get_language_index(
        language,
    )

    resolved_video_filename = resolve_video_display_filename(
        video_path=video_path,
        video_filename=video_filename,
    )

    return (
        gr.update(
            label=create_video_component_label(
                language_index,
                resolved_video_filename,
            ),
        ),
        resolved_video_filename,
    )


def handle_clear_video_filename(
    language: str,
) -> tuple[Any, None]:
    """Restore the default video label and clear the filename state."""

    language_index = get_language_index(
        language,
    )

    return (
        gr.update(
            label=create_video_component_label(
                language_index,
                None,
            ),
        ),
        None,
    )


__all__ = [
    "RECORDED_VIDEO_FILENAME",
    "create_video_component_label",
    "get_video_display_filename",
    "handle_clear_video_filename",
    "handle_video_filename_change",
    "resolve_video_display_filename",
]
