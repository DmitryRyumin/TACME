"""
File: keyframes.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Key-video-moment UI event helpers for the TACME application.

License: MIT License
"""

from typing import Any

import gradio as gr

from tacme.localization import (
    get_language_index,
    get_localized_text,
)
from tacme.visualization.keyframes import (
    KeyVideoMoments,
    create_key_video_moments_html,
)

KEY_VIDEO_MOMENTS_JS = """
() => {
    if (window.__tacmeKeyVideoMomentsBound) {
        return [];
    }

    const selectMoment = (root, momentIndex) => {
        root
            .querySelectorAll(
                ".key-video-moment-card, " +
                ".key-video-timeline-marker"
            )
            .forEach((element) => {
                element.classList.remove(
                    "is-selected"
                );
            });

        root
            .querySelectorAll(
                `[data-moment-index="${momentIndex}"]`
            )
            .forEach((element) => {
                element.classList.add(
                    "is-selected"
                );
            });

        const selectedCard =
            root.querySelector(
                `.key-video-moment-card` +
                `[data-moment-index="${momentIndex}"]`
            );

        selectedCard?.scrollIntoView({
            behavior: "smooth",
            block: "nearest",
            inline: "center",
        });
    };

    document.addEventListener(
        "click",
        (event) => {
            if (!(event.target instanceof Element)) {
                return;
            }

            const trigger =
                event.target.closest(
                    ".key-video-moment-card, " +
                    ".key-video-timeline-marker"
                );

            if (!trigger) {
                return;
            }

            const root =
                trigger.closest(
                    ".key-video-moments-content"
                );

            const momentIndex =
                trigger.getAttribute(
                    "data-moment-index"
                );

            if (!root || momentIndex === null) {
                return;
            }

            event.preventDefault();

            selectMoment(
                root,
                momentIndex
            );
        }
    );

    window.__tacmeKeyVideoMomentsBound = true;

    return [];
}
"""


def _create_key_video_moments_title(
    language_index: int,
) -> str:
    """Create the localized key-video-moments heading."""

    return "### " + get_localized_text(
        "Labels_KEY_VIDEO_MOMENTS",
        language_index,
    )


def handle_render_key_video_moments(
    key_moments: KeyVideoMoments | None,
    language: str,
) -> tuple[Any, Any, Any]:
    """Render representative video moments after successful analysis."""

    language_index = get_language_index(
        language,
    )

    if key_moments is None or not key_moments.moments:
        return (
            gr.update(
                visible=False,
            ),
            gr.update(
                value=_create_key_video_moments_title(
                    language_index,
                ),
            ),
            gr.update(
                value="",
            ),
        )

    return (
        gr.update(
            visible=True,
        ),
        gr.update(
            value=_create_key_video_moments_title(
                language_index,
            ),
        ),
        gr.update(
            value=create_key_video_moments_html(
                key_moments,
                language_index,
            ),
        ),
    )


def handle_localize_key_video_moments(
    key_moments: KeyVideoMoments | None,
    language: str,
) -> tuple[Any, Any, Any]:
    """Relocalize cached key-video-moment UI without rerunning inference."""

    return handle_render_key_video_moments(
        key_moments,
        language,
    )


def handle_clear_key_video_moments() -> tuple[Any, Any, Any, Any]:
    """Clear cached key-video-moment state and hide its UI."""

    return (
        None,
        gr.update(
            visible=False,
        ),
        gr.update(),
        gr.update(
            value="",
        ),
    )


__all__ = [
    "KEY_VIDEO_MOMENTS_JS",
    "handle_clear_key_video_moments",
    "handle_localize_key_video_moments",
    "handle_render_key_video_moments",
]
