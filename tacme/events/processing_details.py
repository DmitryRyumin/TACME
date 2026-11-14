"""Gradio event handlers for the completed TACME processing-details dialog.

Keep this module separate from tacme.ui.processing_details, which builds HTML.
"""

from math import isfinite
from typing import Any

import gradio as gr

from tacme.localization import get_localized_text
from tacme.performance import AnalysisTiming
from tacme.ui.processing_details import create_processing_details_html

_DEFAULT_TEXT = {
    "Texts_DETAILS_TITLE": ("Processing details", "Подробности обработки"),
    "Texts_DETAILS_BUTTON": ("Processing details", "Подробности обработки"),
}


def _language_index(language: str | int) -> int:
    """Convert Gradio's selected language into the existing localization index."""
    if isinstance(language, int):
        return 1 if language == 1 else 0
    value = str(language).strip().lower()
    if value in {"1", "ru", "ru-ru", "russian", "русский"}:
        return 1
    return 0


def _label(key: str, language_index: int) -> str:
    """Prefer the project's translations; preserve a fallback for older catalogs."""
    fallback = _DEFAULT_TEXT[key][language_index]
    try:
        value = get_localized_text(key, language_index)
    except KeyError, LookupError, ValueError:
        return fallback
    if not isinstance(value, str) or not value.strip() or value == key:
        return fallback
    return value


def _is_available(timing: AnalysisTiming | None, is_analysis_running: bool) -> bool:
    return timing is not None and not is_analysis_running


def _button_label(timing: AnalysisTiming | None, language_index: int) -> str:
    """Keep the completed run's measured time, RTF, and device in the button."""
    label = _label("Texts_DETAILS_BUTTON", language_index)
    if timing is None:
        return label

    parts = [label]
    if isfinite(timing.analysis_seconds) and timing.analysis_seconds >= 0.0:
        seconds = f"{timing.analysis_seconds:.2f}"
        if language_index == 1:
            seconds = seconds.replace(".", ",")
        seconds_unit = get_localized_text("Texts_PROGRESS_SECONDS", language_index)
        parts.append(f"{seconds} {seconds_unit}")

    if timing.rtf is not None and isfinite(timing.rtf):
        rtf_text = f"{timing.rtf:.3f}"
        if language_index == 1:
            rtf_text = rtf_text.replace(".", ",")
        parts.append(f"RTF {rtf_text}")

    if timing.device:
        parts.append(str(timing.device))

    return " · ".join(parts)


def handle_open_processing_details(
    analysis_timing: AnalysisTiming | None,
    language: str | int,
) -> tuple[Any, Any, Any]:
    """Open the report for the latest completed analysis, without running models."""
    language_index = _language_index(language)
    available = analysis_timing is not None
    report_html = create_processing_details_html(analysis_timing, language_index) if available else ""
    return (
        gr.update(visible=available),
        gr.update(value=f"### {_label('Texts_DETAILS_TITLE', language_index)}"),
        gr.update(value=report_html),
    )


def handle_close_processing_details() -> Any:
    """Hide the dialog while retaining cached analysis timings."""
    return gr.update(visible=False)


def handle_details_availability(
    analysis_timing: AnalysisTiming | None,
    language: str | int,
    is_analysis_running: bool,
) -> tuple[Any, Any, Any]:
    """Enable the report only for a completed, non-running analysis."""
    language_index = _language_index(language)
    available = _is_available(analysis_timing, is_analysis_running)
    report_html = create_processing_details_html(analysis_timing, language_index) if available else ""
    return (
        gr.update(
            visible=available,
            value=_button_label(analysis_timing, language_index),
        ),
        gr.update() if available else gr.update(visible=False),
        gr.update(value=report_html),
    )


def handle_localize_processing_details(
    analysis_timing: AnalysisTiming | None,
    language: str | int,
    is_analysis_running: bool,
) -> tuple[Any, Any, Any]:
    """Translate button/title/report when the application language changes."""
    language_index = _language_index(language)
    available = _is_available(analysis_timing, is_analysis_running)
    report_html = create_processing_details_html(analysis_timing, language_index) if available else ""
    return (
        gr.update(
            visible=available,
            value=_button_label(analysis_timing, language_index),
        ),
        gr.update(value=f"### {_label('Texts_DETAILS_TITLE', language_index)}"),
        gr.update(value=report_html),
    )


__all__ = [
    "handle_close_processing_details",
    "handle_details_availability",
    "handle_localize_processing_details",
    "handle_open_processing_details",
]
