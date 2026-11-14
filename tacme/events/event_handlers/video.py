"""
File: video.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Video filename callbacks used by Gradio input events.

License: MIT License
"""

from typing import Any

from tacme.events.video_state import (
    RECORDED_VIDEO_FILENAME,
    get_video_display_filename,
    handle_video_filename_change,
)


def handle_file_video_filename_change_event(
    video_path: str | None,
    language: str,
) -> tuple[Any, str | None]:
    """Update the label for a file-backed or example video."""

    return handle_video_filename_change(
        video_path=video_path,
        language=language,
        video_filename=get_video_display_filename(
            video_path,
        ),
    )


def handle_recorded_video_filename_change_event(
    video_path: str | None,
    language: str,
) -> tuple[Any, str | None]:
    """Update the label for a browser-recorded video."""

    return handle_video_filename_change(
        video_path=video_path,
        language=language,
        video_filename=RECORDED_VIDEO_FILENAME,
    )
