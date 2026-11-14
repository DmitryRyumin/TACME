"""
File: processing_details.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Gradio event bindings for the Processing Details modal.

License: MIT License
"""

from gradio.events import Dependency

from tacme.events.processing_details import (
    handle_close_processing_details,
    handle_details_availability,
    handle_localize_processing_details,
    handle_open_processing_details,
)
from tacme.ui.application import ApplicationTabComponents
from tacme.ui.language_selector import LanguageSelectorComponents


def register_processing_details_events(
    language_change_event: Dependency,
    application_content: ApplicationTabComponents,
    language_selector: LanguageSelectorComponents,
) -> None:
    """Register Processing Details bindings in their original order."""

    # Localize the post-run action independently of the checkbox values.
    language_change_event.then(
        fn=handle_localize_processing_details,
        inputs=[
            application_content.analysis_timing_state,
            language_selector.dropdown,
            application_content.analysis_running_state,
        ],
        outputs=[
            application_content.processing_details_button,
            application_content.processing_details_title,
            application_content.processing_details_content,
        ],
        queue=False,
        show_progress="hidden",
    )

    # State resets on video change / Clear; on completion it carries the measured stages.
    application_content.analysis_timing_state.change(
        fn=handle_details_availability,
        inputs=[
            application_content.analysis_timing_state,
            language_selector.dropdown,
            application_content.analysis_running_state,
        ],
        outputs=[
            application_content.processing_details_button,
            application_content.processing_details_modal,
            application_content.processing_details_content,
        ],
        queue=False,
        show_progress="hidden",
    )

    application_content.analysis_running_state.change(
        fn=handle_details_availability,
        inputs=[
            application_content.analysis_timing_state,
            language_selector.dropdown,
            application_content.analysis_running_state,
        ],
        outputs=[
            application_content.processing_details_button,
            application_content.processing_details_modal,
            application_content.processing_details_content,
        ],
        queue=False,
        show_progress="hidden",
    )

    application_content.processing_details_button.click(
        fn=handle_open_processing_details,
        inputs=[
            application_content.analysis_timing_state,
            language_selector.dropdown,
        ],
        outputs=[
            application_content.processing_details_modal,
            application_content.processing_details_title,
            application_content.processing_details_content,
        ],
        queue=False,
        show_progress="hidden",
    )

    application_content.processing_details_close_button.click(
        fn=handle_close_processing_details,
        inputs=[],
        outputs=[application_content.processing_details_modal],
        queue=False,
        show_progress="hidden",
    )
