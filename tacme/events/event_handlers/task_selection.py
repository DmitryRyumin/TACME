"""
File: task_selection.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Gradio event bindings for task selection and dependent summaries.

License: MIT License
"""

from typing import Any

from tacme.events.analysis_summary import handle_render_analysis_summary
from tacme.events.application import handle_task_selection_change
from tacme.events.attribution import (
    FEATURE_ATTRIBUTION_RESIZE_JS,
    handle_render_feature_attribution,
)
from tacme.ui.application import ApplicationTabComponents
from tacme.ui.language_selector import LanguageSelectorComponents


def register_task_selection_events(
    application_content: ApplicationTabComponents,
    language_selector: LanguageSelectorComponents,
    feature_attribution_outputs: list[Any],
    analysis_summary_outputs: list[Any],
) -> None:
    """Register task selection and its dependent result refresh callbacks."""

    task_selection_event = application_content.task_selector.input(
        fn=handle_task_selection_change,
        inputs=[
            application_content.task_selector,
            application_content.source_video_state,
            language_selector.dropdown,
            application_content.result_cache_state,
            application_content.analysis_timing_state,
            application_content.analysis_running_state,
        ],
        outputs=[
            application_content.run_button,
            application_content.status,
            application_content.results_container,
            application_content.emotion_result,
            application_content.personality_result,
            application_content.ambivalence_result,
        ],
        queue=False,
        show_progress="hidden",
    )

    task_attribution_event = task_selection_event.then(
        fn=handle_render_feature_attribution,
        inputs=[
            application_content.feature_attribution_state,
            application_content.task_selector,
            language_selector.dropdown,
            application_content.feature_attribution_task,
            application_content.feature_attribution_target,
            application_content.feature_attribution_view,
        ],
        outputs=feature_attribution_outputs,
        queue=False,
        preprocess=False,  # Radio choices may change while this event is pending.
        show_progress="hidden",
    )

    task_attribution_event.then(
        fn=None,
        inputs=[],
        outputs=[],
        js=FEATURE_ATTRIBUTION_RESIZE_JS,
        queue=False,
        show_progress="hidden",
    )

    task_selection_event.then(
        fn=handle_render_analysis_summary,
        inputs=[
            application_content.result_cache_state,
            application_content.task_selector,
            language_selector.dropdown,
        ],
        outputs=analysis_summary_outputs,
        queue=False,
        show_progress="hidden",
    )
