"""
File: transcription.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Speech transcription UI rendering for the TACME Gradio application.

License: MIT License
"""

from html import escape
from typing import Any

import gradio as gr

from tacme.localization import get_language_index, get_localized_text
from tacme.transcription import WhisperTranscription

SPEECH_TRANSCRIPTION_JS = r"""
() => {
    if (window.__tacmeSpeechTranscriptionCopyHandler) {
        return [];
    }

    const handler = async (event) => {
        if (!(event.target instanceof Element)) {
            return;
        }

        const button = event.target.closest(
            "button.speech-transcription-copy-button"
        );

        if (!button) {
            return;
        }

        const card = button.closest(
            "section.speech-transcription-content"
        );

        const transcript = card?.querySelector(
            ".speech-transcription-text"
        );

        const text = transcript?.textContent?.trim() ?? "";

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
    window.__tacmeSpeechTranscriptionCopyHandler = handler;

    return [];
}
"""


def _format_language(
    transcription: WhisperTranscription,
    language_index: int,
) -> str:
    """Return the language display value used by the transcription card."""

    code = transcription.language_code
    name = transcription.language_name

    if name and code:
        if language_index == 1:
            localized_common = {
                "en": "Английский",
                "ru": "Русский",
            }.get(code)

            if localized_common is not None:
                name = localized_common

        return f"{name} ({code})"

    if name:
        return name

    if code:
        return code.upper()

    return get_localized_text(
        "Labels_TRANSCRIPTION_UNAVAILABLE",
        language_index,
    )


def create_speech_transcription_html(
    transcription: WhisperTranscription,
    language_index: int,
) -> str:
    """Create the complete localized transcription-card HTML."""

    language_label = escape(
        get_localized_text(
            "Labels_TRANSCRIPTION_LANGUAGE",
            language_index,
        )
    )

    confidence_label = escape(
        get_localized_text(
            "Labels_TRANSCRIPTION_LANGUAGE_CONFIDENCE",
            language_index,
        )
    )

    transcript_label = escape(
        get_localized_text(
            "Labels_TRANSCRIPT",
            language_index,
        )
    )

    copy_label = escape(
        get_localized_text(
            "Labels_TRANSCRIPTION_COPY",
            language_index,
        )
    )

    copied_label = escape(
        get_localized_text(
            "Labels_TRANSCRIPTION_COPIED",
            language_index,
        )
    )

    language_value = escape(
        _format_language(
            transcription,
            language_index,
        )
    )

    confidence = transcription.language_confidence

    if confidence is None:
        confidence_value = escape(
            get_localized_text(
                "Labels_TRANSCRIPTION_UNAVAILABLE",
                language_index,
            )
        )
        confidence_percent = 0.0
        confidence_class = "is-unavailable"
    else:
        confidence_percent = max(
            0.0,
            min(
                confidence * 100.0,
                100.0,
            ),
        )
        confidence_value = f"{confidence_percent:.1f}%"
        confidence_class = ""

    transcript = escape(
        transcription.text,
    )

    return f"""
<section class="speech-transcription-content">
    <div class="speech-transcription-metadata">
        <div class="speech-transcription-language-block">
            <span class="speech-transcription-meta-label">{language_label}</span>
            <strong class="speech-transcription-language-value">{language_value}</strong>
        </div>
        <div class="speech-transcription-confidence-block {confidence_class}">
            <div class="speech-transcription-confidence-header">
                <span class="speech-transcription-meta-label">{confidence_label}</span>
                <strong>{confidence_value}</strong>
            </div>
            <div class="speech-transcription-confidence-track" aria-hidden="true">
                <span
                    class="speech-transcription-confidence-fill"
                    style="--tacme-transcription-confidence: {confidence_percent:.3f}%"
                ></span>
            </div>
        </div>
    </div>
    <div class="speech-transcription-body">
        <div class="speech-transcription-toolbar">
            <span>{transcript_label}</span>
            <button
                type="button"
                class="speech-transcription-copy-button"
                data-copy-label="{copy_label}"
                data-copied-label="{copied_label}"
            >{copy_label}</button>
        </div>
        <div class="speech-transcription-scroll">
            <p class="speech-transcription-text">{transcript}</p>
        </div>
    </div>
</section>
""".strip()


def handle_render_speech_transcription(
    transcription: WhisperTranscription | None,
    language: str,
) -> tuple[Any, Any, Any]:
    """Render the retained Whisper transcription after TACME inference."""

    language_index = get_language_index(
        language,
    )

    if transcription is None or not transcription.text.strip():
        return (
            gr.update(
                visible=False,
            ),
            gr.update(
                value=(
                    "### "
                    + get_localized_text(
                        "Labels_SPEECH_TRANSCRIPTION",
                        language_index,
                    )
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
            value=(
                "### "
                + get_localized_text(
                    "Labels_SPEECH_TRANSCRIPTION",
                    language_index,
                )
            ),
        ),
        gr.update(
            value=create_speech_transcription_html(
                transcription,
                language_index,
            ),
        ),
    )


def handle_localize_speech_transcription(
    transcription: WhisperTranscription | None,
    language: str,
) -> tuple[Any, Any, Any]:
    """Relocalize a cached transcription without rerunning Whisper."""

    return handle_render_speech_transcription(
        transcription,
        language,
    )


def handle_clear_speech_transcription() -> tuple[Any, Any, Any, Any]:
    """Reset the transcription state and hide the transcription card."""

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
    "SPEECH_TRANSCRIPTION_JS",
    "create_speech_transcription_html",
    "handle_clear_speech_transcription",
    "handle_localize_speech_transcription",
    "handle_render_speech_transcription",
]
