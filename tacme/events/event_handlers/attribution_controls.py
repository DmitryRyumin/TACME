"""
File: attribution_controls.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Registration of Feature Attribution task, target and view controls.

License: MIT License
"""

from tacme.events.attribution import (
    FEATURE_ATTRIBUTION_RESIZE_JS,
    handle_feature_attribution_target_change,
    handle_feature_attribution_task_change,
    handle_feature_attribution_view_change,
)
from tacme.ui.application import ApplicationTabComponents
from tacme.ui.language_selector import LanguageSelectorComponents


def register_attribution_control_events(
    application_content: ApplicationTabComponents,
    language_selector: LanguageSelectorComponents,
) -> None:
    """Bind Feature Attribution controls and dependent Plotly resize callbacks."""

    attribution_task_event = application_content.feature_attribution_task.input(
        fn=handle_feature_attribution_task_change,
        inputs=[
            application_content.feature_attribution_state,
            application_content.feature_attribution_task,
            application_content.feature_attribution_view,
            language_selector.dropdown,
        ],
        outputs=[
            application_content.feature_attribution_target,
            application_content.feature_attribution_plot,
            application_content.feature_attribution_summary,
        ],
        queue=False,
        preprocess=False,  # Radio choices are populated dynamically.
        show_progress="hidden",
    )

    attribution_task_event.then(
        fn=None,
        inputs=[],
        outputs=[],
        js=FEATURE_ATTRIBUTION_RESIZE_JS,
        queue=False,
        show_progress="hidden",
    )

    attribution_target_event = application_content.feature_attribution_target.input(
        fn=handle_feature_attribution_target_change,
        inputs=[
            application_content.feature_attribution_state,
            application_content.feature_attribution_target,
            application_content.feature_attribution_view,
            language_selector.dropdown,
        ],
        outputs=[
            application_content.feature_attribution_plot,
            application_content.feature_attribution_summary,
        ],
        queue=False,
        preprocess=False,  # Radio choices are populated dynamically.
        show_progress="hidden",
    )

    attribution_target_event.then(
        fn=None,
        inputs=[],
        outputs=[],
        js=FEATURE_ATTRIBUTION_RESIZE_JS,
        queue=False,
        show_progress="hidden",
    )

    attribution_view_event = application_content.feature_attribution_view.input(
        fn=handle_feature_attribution_view_change,
        inputs=[
            application_content.feature_attribution_state,
            application_content.feature_attribution_target,
            application_content.feature_attribution_view,
            language_selector.dropdown,
        ],
        outputs=[
            application_content.feature_attribution_plot,
            application_content.feature_attribution_summary,
        ],
        queue=False,
        preprocess=False,  # Radio choices are populated dynamically.
        show_progress="hidden",
    )

    attribution_view_event.then(
        fn=None,
        inputs=[],
        outputs=[],
        js=FEATURE_ATTRIBUTION_RESIZE_JS,
        queue=False,
        show_progress="hidden",
    )
