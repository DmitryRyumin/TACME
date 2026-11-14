"""
File: video_filename.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Resolve a video display name without registering video-input events.

License: MIT License
"""

from typing import Literal

from tacme.events.event_handlers.video import (
    handle_file_video_filename_change_event,
    handle_recorded_video_filename_change_event,
)

type VideoSource = Literal["file", "webcam", "example", "clear"]


def resolve_video_filename(
    video_path: str | None,
    language: str,
    *,
    source: VideoSource,
) -> str | None:
    """Return the original File/Examples/Webcam display name as plain state.

    The event coordinator applies the label separately from the video value.
    Filename-only updates must never be registered as video-input listeners.
    """

    if not video_path:
        return None
    if source == "webcam":
        _, filename = handle_recorded_video_filename_change_event(video_path, language)
    else:
        _, filename = handle_file_video_filename_change_event(video_path, language)
    return filename
