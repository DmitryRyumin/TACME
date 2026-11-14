"""
File: clear.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Atomic Gradio Clear event without duplicate downstream callbacks.

License: MIT License
"""

from typing import Any

import gradio as gr

from tacme.events.application import handle_clear_application
from tacme.events.event_handlers.video_reset import create_video_reset_updates
from tacme.events.media_info import handle_media_change
from tacme.events.video_state import create_video_component_label
from tacme.localization import get_language_index
from tacme.ui.application import ApplicationTabComponents
from tacme.ui.client_scripts import CLEAR_EXAMPLE_SELECTION_JS
from tacme.ui.language_selector import LanguageSelectorComponents


def handle_clear_transition(language: str) -> tuple[Any, ...]:
    """Reset the video, filename, metadata and all results in one transaction."""

    application = handle_clear_application(language)
    media_info = handle_media_change(None, language)

    return (
        gr.update(
            value=None,
            label=create_video_component_label(get_language_index(language), None),
        ),
        *application[1:],
        None,  # source_video_state
        None,  # video_filename_state
        *media_info[:3],
        *media_info[4:],  # media modal is already reset by handle_clear_application
        *create_video_reset_updates(),
    )


def register_clear_events(
    application_content: ApplicationTabComponents,
    language_selector: LanguageSelectorComponents,
    media_info_outputs: list[Any],
    key_video_moments_outputs: list[Any],
    audio_waveform_outputs: list[Any],
    speech_transcription_outputs: list[Any],
    behavior_description_outputs: list[Any],
    feature_attribution_outputs: list[Any],
    analysis_summary_outputs: list[Any],
) -> None:
    """Register exactly one server event for an explicit Clear/Got it click."""

    application_content.clear_button.click(
        fn=handle_clear_transition,
        inputs=[language_selector.dropdown],
        outputs=[
            application_content.video_input,
            application_content.task_selector,
            application_content.run_button,
            application_content.clear_button,
            application_content.status,
            application_content.results_container,
            application_content.emotion_result,
            application_content.personality_result,
            application_content.ambivalence_result,
            application_content.result_cache_state,
            application_content.analysis_timing_state,
            application_content.analysis_busy_overlay,
            application_content.analysis_running_state,
            application_content.processing_details_button,
            application_content.processing_details_modal,
            application_content.processing_details_content,
            application_content.media_info_modal,
            application_content.source_video_state,
            application_content.video_filename_state,
            *media_info_outputs[:3],
            *media_info_outputs[4:],
            *key_video_moments_outputs,
            *audio_waveform_outputs,
            *speech_transcription_outputs,
            *behavior_description_outputs,
            *feature_attribution_outputs,
            *analysis_summary_outputs,
        ],
        js=CLEAR_EXAMPLE_SELECTION_JS,
        queue=False,
        show_progress="hidden",
    )
