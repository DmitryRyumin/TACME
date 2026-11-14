"""
File: controls.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Video changes, task-selection changes and clear-button UI state updates.

License: MIT License
"""

from typing import Any

import gradio as gr

from tacme.events.application.results import create_result_updates
from tacme.events.application.status import (
    _video_file_status_text,
    create_analysis_completed_status,
)
from tacme.events.video_state import create_video_component_label
from tacme.inference import TacmeInferenceResult
from tacme.localization import (
    TASK_KEYS,
    get_language_index,
    get_localized_text,
    get_task_choices,
    get_task_keys_from_labels,
)
from tacme.media.validation import VideoFileStatus, check_video_container
from tacme.performance import AnalysisTiming


def handle_video_change(
    video_path: str | None,
    selected_tasks: list[str] | None,
    language: str,
    *,
    video_status: VideoFileStatus | None = None,
    video_filename: str | None = None,
) -> tuple[Any, ...]:
    """Update TACME controls for a single inspected video selection."""

    language_index = get_language_index(
        language,
    )

    task_choices = get_task_choices(language_index)
    all_task_keys = list(TASK_KEYS)

    if video_status is None:
        video_status = check_video_container(video_path)
    if video_status != "available":
        return (
            gr.update(
                choices=task_choices,
                value=all_task_keys,
                visible=False,
                interactive=True,
            ),
            gr.update(
                interactive=False,
            ),
            gr.update(
                interactive=bool(video_path),
            ),
            gr.update(
                value=_video_file_status_text(video_status, language_index),
            ),
            gr.update(
                visible=False,
            ),
            gr.update(
                value="",
                visible=False,
            ),
            gr.update(
                value="",
                visible=False,
            ),
            gr.update(
                value="",
                visible=False,
            ),
            None,
            None,
            "",
            False,
            gr.update(
                visible=False,
                value=get_localized_text(
                    "Texts_DETAILS_BUTTON",
                    language_index,
                ),
            ),
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(
                value=None,
                label=create_video_component_label(language_index, None),
            )
            if video_path
            else gr.update(
                label=create_video_component_label(language_index, None),
            ),
        )

    selected = get_task_keys_from_labels(selected_tasks) if selected_tasks is not None else all_task_keys

    has_tasks = bool(
        selected,
    )

    status_text = get_localized_text(
        ("Texts_STATUS_VIDEO_READY" if has_tasks else "Texts_STATUS_NO_TASKS"),
        language_index,
    )

    return (
        gr.update(
            choices=task_choices,
            value=selected,
            visible=True,
            interactive=True,
        ),
        gr.update(
            interactive=has_tasks,
        ),
        gr.update(
            interactive=True,
        ),
        gr.update(
            value=status_text,
        ),
        gr.update(
            visible=False,
        ),
        gr.update(
            value="",
            visible=False,
        ),
        gr.update(
            value="",
            visible=False,
        ),
        gr.update(
            value="",
            visible=False,
        ),
        None,
        None,
        "",
        False,
        gr.update(
            visible=False,
            value=get_localized_text(
                "Texts_DETAILS_BUTTON",
                language_index,
            ),
        ),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(
            label=create_video_component_label(language_index, video_filename),
        ),
    )


def handle_task_selection_change(
    selected_tasks: list[str] | None,
    video_path: str | None,
    language: str,
    cached_result: TacmeInferenceResult | None,
    analysis_timing: AnalysisTiming | None,
    is_analysis_running: bool,
) -> tuple[Any, ...]:
    """Update controls and cached TACME results after task selection changes."""

    language_index = get_language_index(
        language,
    )

    if is_analysis_running:
        return (
            gr.update(
                interactive=False,
            ),
            gr.update(
                value=get_localized_text(
                    "Texts_STATUS_ANALYSIS_RUNNING",
                    language_index,
                ),
            ),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    video_status = check_video_container(video_path)
    if video_status != "available":
        return (
            gr.update(
                interactive=False,
            ),
            gr.update(
                value=_video_file_status_text(video_status, language_index),
            ),
            gr.update(
                visible=False,
            ),
            gr.update(
                visible=False,
            ),
            gr.update(
                visible=False,
            ),
            gr.update(
                visible=False,
            ),
        )

    if not selected_tasks:
        return (
            gr.update(
                interactive=False,
            ),
            gr.update(
                value=get_localized_text(
                    "Texts_STATUS_NO_TASKS",
                    language_index,
                ),
            ),
            gr.update(
                visible=False,
            ),
            gr.update(
                visible=False,
            ),
            gr.update(
                visible=False,
            ),
            gr.update(
                visible=False,
            ),
        )

    if cached_result is not None:
        (
            results_container,
            emotion_result,
            personality_result,
            ambivalence_result,
        ) = create_result_updates(
            cached_result,
            selected_tasks,
            language_index,
        )

        return (
            gr.update(
                interactive=True,
            ),
            gr.update(
                value=create_analysis_completed_status(
                    analysis_timing,
                    language_index,
                ),
            ),
            results_container,
            emotion_result,
            personality_result,
            ambivalence_result,
        )

    return (
        gr.update(
            interactive=True,
        ),
        gr.update(
            value=get_localized_text(
                "Texts_STATUS_VIDEO_READY",
                language_index,
            ),
        ),
        gr.update(
            visible=False,
        ),
        gr.update(
            visible=False,
        ),
        gr.update(
            visible=False,
        ),
        gr.update(
            visible=False,
        ),
    )


def handle_clear_application(
    language: str,
) -> tuple[Any, ...]:
    """Reset the TACME application tab to its initial state."""

    language_index = get_language_index(
        language,
    )

    task_choices = get_task_choices(language_index)

    return (
        None,
        gr.update(
            choices=task_choices,
            value=list(TASK_KEYS),
            visible=False,
            interactive=True,
        ),
        gr.update(
            interactive=False,
        ),
        gr.update(
            interactive=False,
        ),
        gr.update(
            value=get_localized_text(
                "Texts_STATUS_CLEARED",
                language_index,
            ),
        ),
        gr.update(
            visible=False,
        ),
        gr.update(
            value="",
            visible=False,
        ),
        gr.update(
            value="",
            visible=False,
        ),
        gr.update(
            value="",
            visible=False,
        ),
        None,
        None,
        "",
        False,
        gr.update(
            visible=False,
            value=get_localized_text(
                "Texts_DETAILS_BUTTON",
                language_index,
            ),
        ),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(visible=False),  # Close Media Information immediately on Clear.
    )
