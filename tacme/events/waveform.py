"""
File: waveform.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Audio-waveform event handlers for the TACME Gradio application.

License: MIT License
"""

from typing import Any

import gradio as gr

from tacme.inference import TacmeInferenceResult
from tacme.localization import (
    get_language_index,
    get_localized_text,
)
from tacme.visualization.waveform import (
    AudioWaveformEnvelope,
    create_audio_waveform_envelope,
    create_audio_waveform_figure,
)

AUDIO_WAVEFORM_RESIZE_JS = r"""
() => {
    const resizeWaveform = () => {
        const root = document.getElementById(
            "application-audio-waveform-plot"
        );

        const plot = root?.querySelector(
            ".js-plotly-plot"
        );

        if (plot && window.Plotly?.Plots?.resize) {
            window.Plotly.Plots.resize(plot);
        }

        window.dispatchEvent(new Event("resize"));
    };

    window.requestAnimationFrame(() => {
        window.requestAnimationFrame(resizeWaveform);
    });

    window.setTimeout(resizeWaveform, 80);
    window.setTimeout(resizeWaveform, 220);

    return [];
}
"""


def _create_waveform_title(
    language_index: int,
) -> str:
    """Create the localized audio-waveform section title."""

    return "### " + get_localized_text(
        "Labels_AUDIO_WAVEFORM",
        language_index,
    )


def handle_clear_audio_waveform() -> tuple[Any, ...]:
    """Clear and hide the audio-waveform visualization."""

    return (
        None,
        gr.update(
            visible=False,
        ),
        gr.update(),
        gr.update(
            value=None,
        ),
    )


def handle_create_audio_waveform(
    video_path: str | None,
    language: str,
    cached_result: TacmeInferenceResult | None,
) -> tuple[Any, ...]:
    """Create the waveform only after a successful TACME analysis."""

    if not video_path or cached_result is None:
        return handle_clear_audio_waveform()

    language_index = get_language_index(
        language,
    )

    try:
        waveform = create_audio_waveform_envelope(
            video_path,
        )
        figure = create_audio_waveform_figure(
            waveform,
            language_index,
        )

    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
    ):
        return handle_clear_audio_waveform()

    return (
        waveform,
        gr.update(
            visible=True,
        ),
        gr.update(
            value=_create_waveform_title(
                language_index,
            ),
        ),
        gr.update(
            value=figure,
        ),
    )


def handle_localize_audio_waveform(
    waveform: AudioWaveformEnvelope | None,
    language: str,
) -> tuple[Any, ...]:
    """Relocalize an already decoded waveform without decoding media again."""

    language_index = get_language_index(
        language,
    )

    if waveform is None:
        return (
            gr.update(
                visible=False,
            ),
            gr.update(
                value=_create_waveform_title(
                    language_index,
                ),
            ),
            gr.update(),
        )

    return (
        gr.update(
            visible=True,
        ),
        gr.update(
            value=_create_waveform_title(
                language_index,
            ),
        ),
        gr.update(
            value=create_audio_waveform_figure(
                waveform,
                language_index,
            ),
        ),
    )


__all__ = [
    "AUDIO_WAVEFORM_RESIZE_JS",
    "handle_clear_audio_waveform",
    "handle_create_audio_waveform",
    "handle_localize_audio_waveform",
]
