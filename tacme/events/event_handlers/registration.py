"""
File: registration.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Event handler registration for the TACME Gradio application.

License: MIT License
"""

from typing import cast

import gradio as gr

from tacme.events.event_handlers.attribution_controls import (
    register_attribution_control_events,
)
from tacme.events.event_handlers.bootstrap import register_client_bootstrap_events
from tacme.events.event_handlers.clear import register_clear_events
from tacme.events.event_handlers.language import create_language_change_handler
from tacme.events.event_handlers.localization import register_localization_events
from tacme.events.event_handlers.media_info import register_media_info_modal_events
from tacme.events.event_handlers.processing_details import register_processing_details_events
from tacme.events.event_handlers.requirements import register_requirements_events
from tacme.events.event_handlers.run_analysis import register_run_analysis_events
from tacme.events.event_handlers.task_selection import register_task_selection_events
from tacme.events.event_handlers.video_validation import register_video_validation_events
from tacme.media.upload_logging import install_upload_disconnect_log_filter
from tacme.ui.about import (
    AboutTabComponents,
)
from tacme.ui.application import (
    ApplicationTabComponents,
)
from tacme.ui.authors import (
    AuthorsTabComponents,
)
from tacme.ui.language_selector import (
    LanguageSelectorComponents,
)
from tacme.ui.requirements import RequirementsTabComponents
from tacme.ui.tabs import (
    AppTabsComponents,
)


def setup_app_event_handlers(
    gradio_app: gr.Blocks,
    language_selector: LanguageSelectorComponents,
    app_tabs: AppTabsComponents,
) -> None:
    """Register TACME application event handlers."""

    # The browser may cancel a multipart upload before TACME receives a file.
    # Hide only the expected Gradio upload-route ClientDisconnect traceback.
    install_upload_disconnect_log_filter()

    application_content = cast(
        ApplicationTabComponents,
        app_tabs.tab_contents["APP"],
    )

    about_content = cast(
        AboutTabComponents,
        app_tabs.tab_contents["ABOUT"],
    )

    authors_content = cast(
        AuthorsTabComponents,
        app_tabs.tab_contents["AUTHORS"],
    )

    requirements_content = cast(
        RequirementsTabComponents,
        app_tabs.tab_contents["REQUIREMENTS"],
    )

    handle_language_change_event = create_language_change_handler(
        language_selector=language_selector,
        app_tabs=app_tabs,
        application_content=application_content,
        about_content=about_content,
        authors_content=authors_content,
        requirements_content=requirements_content,
    )

    language_outputs = [
        language_selector.flag,
        *app_tabs.tab_components.values(),
        application_content.title,
        application_content.video_input,
        application_content.examples_title,
        application_content.examples_placeholder,
        application_content.task_selector,
        application_content.run_button,
        application_content.clear_button,
        application_content.status,
        application_content.media_info_button_column,
        application_content.media_info_button,
        application_content.media_info_modal_title,
        application_content.media_overview_tab,
        application_content.media_video_tab,
        application_content.media_audio_tab,
        application_content.media_compatibility_tab,
        application_content.media_overview_content,
        application_content.media_video_content,
        application_content.media_audio_content,
        application_content.media_compatibility_content,
        application_content.results_container,
        application_content.emotion_result,
        application_content.personality_result,
        application_content.ambivalence_result,
        about_content.title,
        about_content.description,
        about_content.content,
        authors_content.content,
        requirements_content.title,
        requirements_content.description,
        requirements_content.content,
    ]

    register_client_bootstrap_events(gradio_app, language_selector)

    register_requirements_events(gradio_app, language_selector, requirements_content)

    language_change_event = language_selector.dropdown.change(
        fn=handle_language_change_event,
        inputs=[
            language_selector.dropdown,
            requirements_content.session_state,
            application_content.task_selector,
            application_content.source_video_state,
            application_content.video_filename_state,
            application_content.media_info_state,
            application_content.result_cache_state,
            application_content.analysis_timing_state,
            application_content.analysis_running_state,
        ],
        outputs=language_outputs,
        queue=False,
        show_progress="hidden",
    )

    register_processing_details_events(
        language_change_event,
        application_content,
        language_selector,
    )

    register_localization_events(
        language_selector,
        application_content,
        language_change_event,
    )

    key_video_moments_outputs = [
        application_content.key_video_moments_state,
        application_content.key_video_moments_container,
        application_content.key_video_moments_title,
        application_content.key_video_moments_content,
    ]

    audio_waveform_outputs = [
        application_content.audio_waveform_state,
        application_content.audio_waveform_container,
        application_content.audio_waveform_title,
        application_content.audio_waveform_plot,
    ]

    speech_transcription_outputs = [
        application_content.speech_transcription_state,
        application_content.speech_transcription_container,
        application_content.speech_transcription_title,
        application_content.speech_transcription_content,
    ]

    behavior_description_outputs = [
        application_content.behavior_description_state,
        application_content.behavior_description_container,
        application_content.behavior_description_title,
        application_content.behavior_description_content,
    ]

    feature_attribution_outputs = [
        application_content.feature_attribution_state,
        application_content.feature_attribution_container,
        application_content.feature_attribution_title,
        application_content.feature_attribution_task,
        application_content.feature_attribution_target,
        application_content.feature_attribution_view,
        application_content.feature_attribution_plot,
        application_content.feature_attribution_summary,
        application_content.feature_attribution_note,
    ]

    analysis_summary_outputs = [
        application_content.analysis_summary_container,
        application_content.analysis_summary_content,
    ]

    media_info_outputs = [
        application_content.media_info_state,
        application_content.media_info_button_column,
        application_content.media_info_button,
        application_content.media_info_modal,
        application_content.media_overview_content,
        application_content.media_video_content,
        application_content.media_audio_content,
        application_content.media_compatibility_content,
    ]

    register_video_validation_events(
        application_content,
        language_selector,
        media_info_outputs,
        key_video_moments_outputs,
        audio_waveform_outputs,
        speech_transcription_outputs,
        behavior_description_outputs,
        feature_attribution_outputs,
        analysis_summary_outputs,
    )

    register_media_info_modal_events(application_content)

    register_task_selection_events(
        application_content,
        language_selector,
        feature_attribution_outputs,
        analysis_summary_outputs,
    )

    register_run_analysis_events(
        application_content,
        language_selector,
        audio_waveform_outputs,
        feature_attribution_outputs,
        analysis_summary_outputs,
    )
    register_attribution_control_events(
        application_content,
        language_selector,
    )

    register_clear_events(
        application_content,
        language_selector,
        media_info_outputs,
        key_video_moments_outputs,
        audio_waveform_outputs,
        speech_transcription_outputs,
        behavior_description_outputs,
        feature_attribution_outputs,
        analysis_summary_outputs,
    )
