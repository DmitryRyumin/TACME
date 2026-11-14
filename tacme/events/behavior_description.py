"""
File: behavior_description.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Qwen3-VL observable behavior UI rendering for the TACME Gradio application.

License: MIT License
"""

from html import escape
from typing import Any

import gradio as gr

from tacme.behavior import BehaviorDescription
from tacme.localization import get_language_index, get_localized_text

BEHAVIOR_DESCRIPTION_JS = r"""
() => {
    if (window.__tacmeBehaviorDescriptionCopyHandler) {
        return [];
    }

    const handler = async (event) => {
        if (!(event.target instanceof Element)) {
            return;
        }

        const button = event.target.closest(
            "button.behavior-description-copy-button"
        );

        if (!button) {
            return;
        }

        const card = button.closest(
            "section.behavior-description-content"
        );

        const description = card?.querySelector(
            ".behavior-description-text"
        );

        const text = description?.textContent?.trim() ?? "";

        if (!text) {
            return;
        }

        try {
            await navigator.clipboard.writeText(text);
        } catch {
            const textarea = document.createElement("textarea");
            textarea.value = text;
            textarea.style.position = "fixed";
            textarea.style.opacity = "0";
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand("copy");
            textarea.remove();
        }

        const copyLabel = button.dataset.copyLabel ?? "Copy";
        const copiedLabel = button.dataset.copiedLabel ?? "Copied";

        button.textContent = copiedLabel;
        button.classList.add("is-copied");

        window.setTimeout(() => {
            button.textContent = copyLabel;
            button.classList.remove("is-copied");
        }, 1400);
    };

    document.addEventListener("click", handler);
    window.__tacmeBehaviorDescriptionCopyHandler = handler;

    return [];
}
"""


def _format_video_fps(
    behavior: BehaviorDescription,
    language_index: int,
) -> str:
    """Return the localized Qwen3-VL video sampling rate."""

    fps = behavior.video_fps
    fps_value = f"{fps:g}"
    unit = get_localized_text(
        "Units_FRAMES_PER_SECOND",
        language_index,
    )

    return f"{fps_value} {unit}"


def create_behavior_description_html(
    behavior: BehaviorDescription,
    language_index: int,
) -> str:
    """Create the localized observable-behavior card HTML."""

    model_type_label = escape(
        get_localized_text(
            "Labels_BEHAVIOR_MODEL_TYPE",
            language_index,
        )
    )

    description_label = escape(
        get_localized_text(
            "Labels_BEHAVIOR_VISUAL_DESCRIPTION",
            language_index,
        )
    )

    sampling_label = escape(
        get_localized_text(
            "Labels_BEHAVIOR_SAMPLING",
            language_index,
        )
    )

    max_frames_label = escape(
        get_localized_text(
            "Labels_BEHAVIOR_MAX_FRAMES",
            language_index,
        )
    )

    copy_label = escape(
        get_localized_text(
            "Labels_BEHAVIOR_COPY",
            language_index,
        )
    )

    copied_label = escape(
        get_localized_text(
            "Labels_BEHAVIOR_COPIED",
            language_index,
        )
    )

    model_name = escape(
        behavior.model_name.rsplit(
            "/",
            maxsplit=1,
        )[-1]
    )

    sampling = escape(f"{sampling_label}: {_format_video_fps(behavior, language_index)}")

    max_frames = escape(f"{max_frames_label}: {behavior.max_frames}")

    description = escape(
        behavior.text,
    )

    return f"""
<section class="behavior-description-content">
    <div class="behavior-description-metadata">
        <div class="behavior-description-model">
            <span class="behavior-description-meta-label">{model_type_label}</span>
            <strong class="behavior-description-model-pill">{model_name}</strong>
        </div>
        <div class="behavior-description-chips" aria-label="{model_type_label}">
            <span>{sampling}</span>
            <span>{max_frames}</span>
        </div>
    </div>
    <div class="behavior-description-body">
        <div class="behavior-description-toolbar">
            <span>{description_label}</span>
            <button
                type="button"
                class="behavior-description-copy-button"
                data-copy-label="{copy_label}"
                data-copied-label="{copied_label}"
            >{copy_label}</button>
        </div>
        <div class="behavior-description-scroll">
            <p class="behavior-description-text">{description}</p>
        </div>
    </div>
</section>
""".strip()


def handle_render_behavior_description(
    behavior: BehaviorDescription | None,
    language: str,
) -> tuple[Any, Any, Any]:
    """Render the retained Qwen3-VL behavior description after TACME inference."""

    language_index = get_language_index(
        language,
    )

    title = "### " + get_localized_text(
        "Labels_OBSERVED_BEHAVIOR",
        language_index,
    )

    if behavior is None or not behavior.text.strip():
        return (
            gr.update(
                visible=False,
            ),
            gr.update(
                value=title,
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
            value=title,
        ),
        gr.update(
            value=create_behavior_description_html(
                behavior,
                language_index,
            ),
        ),
    )


def handle_localize_behavior_description(
    behavior: BehaviorDescription | None,
    language: str,
) -> tuple[Any, Any, Any]:
    """Relocalize cached behavior metadata without rerunning Qwen3-VL."""

    return handle_render_behavior_description(
        behavior,
        language,
    )


def handle_clear_behavior_description() -> tuple[Any, Any, Any, Any]:
    """Reset the behavior state and hide the observable-behavior card."""

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
    "BEHAVIOR_DESCRIPTION_JS",
    "create_behavior_description_html",
    "handle_clear_behavior_description",
    "handle_localize_behavior_description",
    "handle_render_behavior_description",
]
