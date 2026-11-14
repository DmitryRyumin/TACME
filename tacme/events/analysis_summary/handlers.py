"""
File: handlers.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Gradio handlers for rendering and clearing the analysis summary.

License: MIT License
"""

from typing import Any

import gradio as gr

from tacme.events.analysis_summary.rendering import create_analysis_summary_html
from tacme.inference import TacmeInferenceResult
from tacme.localization import get_language_index, get_task_keys_from_labels


def handle_render_analysis_summary(
    result: TacmeInferenceResult | None,
    selected_tasks: list[str] | None,
    language: str,
) -> tuple[Any, Any]:
    """Render the deterministic summary from cached selected TACME predictions."""

    if result is None or not selected_tasks:
        return (
            gr.update(
                visible=False,
            ),
            gr.update(
                value="",
            ),
        )

    language_index = get_language_index(
        language,
    )

    task_keys = set(
        get_task_keys_from_labels(
            selected_tasks,
        )
    )

    if not task_keys.intersection(
        {
            "emotion",
            "personality",
            "ambivalence",
        }
    ):
        return (
            gr.update(
                visible=False,
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
            value=create_analysis_summary_html(
                result,
                selected_tasks,
                language_index,
            ),
        ),
    )


def handle_clear_analysis_summary() -> tuple[Any, Any]:
    """Clear and hide the analysis-summary section."""

    return (
        gr.update(
            visible=False,
        ),
        gr.update(
            value="",
        ),
    )
