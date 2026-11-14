"""
File: localization.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Gradio localization of application results and visualizations.

License: MIT License
"""

from gradio.events import Dependency

from tacme.events.analysis_summary import handle_render_analysis_summary
from tacme.events.application import handle_localize_predictions_title
from tacme.events.attribution import (
    FEATURE_ATTRIBUTION_RESIZE_JS,
    handle_render_feature_attribution,
)
from tacme.events.behavior_description import handle_localize_behavior_description
from tacme.events.keyframes import handle_localize_key_video_moments
from tacme.events.transcription import handle_localize_speech_transcription
from tacme.events.waveform import AUDIO_WAVEFORM_RESIZE_JS, handle_localize_audio_waveform
from tacme.ui.application import ApplicationTabComponents
from tacme.ui.client_scripts import EXAMPLES_UI_JS
from tacme.ui.language_selector import LanguageSelectorComponents


def register_localization_events(
    language_selector: LanguageSelectorComponents,
    application_content: ApplicationTabComponents,
    language_change_event: Dependency,
) -> None:
    """Register localized UI updates and dependent resize JS in original order."""

    language_selector.dropdown.change(
        fn=None,
        inputs=[
            language_selector.dropdown,
        ],
        outputs=[],
        js=EXAMPLES_UI_JS,
        queue=False,
        show_progress="hidden",
    )

    language_selector.dropdown.change(
        fn=handle_localize_key_video_moments,
        inputs=[
            application_content.key_video_moments_state,
            language_selector.dropdown,
        ],
        outputs=[
            application_content.key_video_moments_container,
            application_content.key_video_moments_title,
            application_content.key_video_moments_content,
        ],
        queue=False,
        show_progress="hidden",
    )

    localized_audio_waveform_event = language_selector.dropdown.change(
        fn=handle_localize_audio_waveform,
        inputs=[
            application_content.audio_waveform_state,
            language_selector.dropdown,
        ],
        outputs=[
            application_content.audio_waveform_container,
            application_content.audio_waveform_title,
            application_content.audio_waveform_plot,
        ],
        queue=False,
        show_progress="hidden",
    )

    localized_audio_waveform_event.then(
        fn=None,
        inputs=[],
        outputs=[],
        js=AUDIO_WAVEFORM_RESIZE_JS,
        queue=False,
        show_progress="hidden",
    )

    language_selector.dropdown.change(
        fn=handle_localize_speech_transcription,
        inputs=[
            application_content.speech_transcription_state,
            language_selector.dropdown,
        ],
        outputs=[
            application_content.speech_transcription_container,
            application_content.speech_transcription_title,
            application_content.speech_transcription_content,
        ],
        queue=False,
        show_progress="hidden",
    )

    language_selector.dropdown.change(
        fn=handle_localize_behavior_description,
        inputs=[
            application_content.behavior_description_state,
            language_selector.dropdown,
        ],
        outputs=[
            application_content.behavior_description_container,
            application_content.behavior_description_title,
            application_content.behavior_description_content,
        ],
        queue=False,
        show_progress="hidden",
    )

    language_selector.dropdown.change(
        fn=handle_localize_predictions_title,
        inputs=[
            language_selector.dropdown,
        ],
        outputs=[
            application_content.predictions_title,
        ],
        queue=False,
        show_progress="hidden",
    )

    localized_feature_attribution_event = language_change_event.then(
        fn=handle_render_feature_attribution,
        inputs=[
            application_content.feature_attribution_state,
            application_content.task_selector,
            language_selector.dropdown,
            application_content.feature_attribution_task,
            application_content.feature_attribution_target,
            application_content.feature_attribution_view,
        ],
        outputs=[
            application_content.feature_attribution_state,
            application_content.feature_attribution_container,
            application_content.feature_attribution_title,
            application_content.feature_attribution_task,
            application_content.feature_attribution_target,
            application_content.feature_attribution_view,
            application_content.feature_attribution_plot,
            application_content.feature_attribution_summary,
            application_content.feature_attribution_note,
        ],
        queue=False,
        preprocess=False,  # Radio choices may change while this event is pending.
        show_progress="hidden",
    )

    localized_feature_attribution_event.then(
        fn=None,
        inputs=[],
        outputs=[],
        js=FEATURE_ATTRIBUTION_RESIZE_JS,
        queue=False,
        show_progress="hidden",
    )

    language_change_event.then(
        fn=handle_render_analysis_summary,
        inputs=[
            application_content.result_cache_state,
            application_content.task_selector,
            language_selector.dropdown,
        ],
        outputs=[
            application_content.analysis_summary_container,
            application_content.analysis_summary_content,
        ],
        queue=False,
        show_progress="hidden",
    )
