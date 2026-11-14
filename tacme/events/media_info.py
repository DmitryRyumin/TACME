"""
File: media_info.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Media-information event handlers for the TACME Gradio application.

License: MIT License
"""

from typing import Any

import gradio as gr

from tacme.localization import (
    get_language_index,
    get_localized_text,
)
from tacme.media import (
    MediaProbeResult,
    create_audio_stream_html,
    create_media_compatibility_html,
    create_media_overview_html,
    create_video_stream_html,
    probe_media_file,
)
from tacme.media.validation import check_video_container


def create_media_content_updates(
    probe_result: MediaProbeResult | None,
    language_index: int,
) -> tuple[Any, Any, Any, Any]:
    """Create localized updates for all media-information tabs."""

    return (
        gr.update(
            value=create_media_overview_html(
                probe_result,
                language_index,
            ),
        ),
        gr.update(
            value=create_video_stream_html(
                probe_result,
                language_index,
            ),
        ),
        gr.update(
            value=create_audio_stream_html(
                probe_result,
                language_index,
            ),
        ),
        gr.update(
            value=create_media_compatibility_html(
                probe_result,
                language_index,
            ),
        ),
    )


def handle_media_change(
    video_path: str | None,
    language: str,
    *,
    verified: bool = False,
) -> tuple[Any, ...]:
    """Prepare Media Information from a previously validated video."""

    # Video-change callbacks may finish after the user opens Media Information.
    # Only the explicit open/close and Clear actions should set modal visibility.

    language_index = get_language_index(
        language,
    )

    if not video_path or (not verified and check_video_container(video_path) != "available"):
        return (
            None,
            gr.update(
                visible=False,
            ),
            gr.update(
                value=get_localized_text(
                    "Labels_MEDIA_INFO_TITLE",
                    language_index,
                ),
                interactive=False,
            ),
            gr.update(),  # Keep the modal's current visibility unchanged.
            gr.update(
                value="",
            ),
            gr.update(
                value="",
            ),
            gr.update(
                value="",
            ),
            gr.update(
                value="",
            ),
        )

    probe_result = probe_media_file(
        video_path,
    )

    if not probe_result.is_available:
        # The file may have changed since the cached container inspection.
        # Expected invalid files should never expose an empty media modal.
        return (
            None,
            gr.update(visible=False),
            gr.update(
                value=get_localized_text("Labels_MEDIA_INFO_TITLE", language_index),
                interactive=False,
            ),
            gr.update(),  # Keep the modal's current visibility unchanged.
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(value=""),
        )

    (
        overview_update,
        video_update,
        audio_update,
        compatibility_update,
    ) = create_media_content_updates(
        probe_result,
        language_index,
    )

    return (
        probe_result,
        gr.update(
            visible=True,
        ),
        gr.update(
            value=get_localized_text(
                "Labels_MEDIA_INFO_TITLE",
                language_index,
            ),
            interactive=True,
        ),
        gr.update(),  # Keep the modal's current visibility unchanged.
        overview_update,
        video_update,
        audio_update,
        compatibility_update,
    )


def handle_show_media_info() -> tuple[Any, Any]:
    """Open the media-information modal on the Overview tab."""

    return (
        gr.update(
            visible=True,
        ),
        gr.update(
            selected="overview",
        ),
    )


def handle_hide_media_info() -> Any:
    """Close the media-information modal."""

    return gr.update(
        visible=False,
    )


__all__ = [
    "create_media_content_updates",
    "handle_hide_media_info",
    "handle_media_change",
    "handle_show_media_info",
]
