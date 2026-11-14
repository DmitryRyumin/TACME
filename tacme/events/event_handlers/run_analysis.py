"""
File: run_analysis.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Gradio event registration for the complete TACME analysis chain.

License: MIT License
"""

from typing import Any

from tacme.events.analysis_summary import handle_render_analysis_summary
from tacme.events.application import handle_analysis_started, handle_run_analysis
from tacme.events.attribution import (
    FEATURE_ATTRIBUTION_RESIZE_JS,
    handle_render_feature_attribution,
)
from tacme.events.behavior_description import handle_render_behavior_description
from tacme.events.keyframes import handle_render_key_video_moments
from tacme.events.transcription import handle_render_speech_transcription
from tacme.events.waveform import AUDIO_WAVEFORM_RESIZE_JS, handle_create_audio_waveform
from tacme.ui.application import ApplicationTabComponents
from tacme.ui.language_selector import LanguageSelectorComponents


def register_run_analysis_events(
    application_content: ApplicationTabComponents,
    language_selector: LanguageSelectorComponents,
    audio_waveform_outputs: list[Any],
    feature_attribution_outputs: list[Any],
    analysis_summary_outputs: list[Any],
) -> None:
    """Register the complete Run Analysis and sequential result-rendering chain."""

    run_analysis_event = application_content.run_button.click(
        fn=handle_analysis_started,
        inputs=[
            application_content.source_video_state,
            language_selector.dropdown,
        ],
        outputs=[
            application_content.run_button,
            application_content.clear_button,
            application_content.task_selector,
            application_content.video_input,
            application_content.status,
            application_content.analysis_busy_overlay,
            application_content.analysis_running_state,
            application_content.processing_details_button,
        ],
        queue=False,
        show_progress="hidden",
    )

    analysis_completed_event = run_analysis_event.then(
        fn=handle_run_analysis,
        inputs=[
            application_content.source_video_state,
            application_content.task_selector,
            language_selector.dropdown,
            application_content.media_info_state,
        ],
        outputs=[
            application_content.status,
            application_content.results_container,
            application_content.emotion_result,
            application_content.personality_result,
            application_content.ambivalence_result,
            application_content.result_cache_state,
            application_content.analysis_timing_state,
            application_content.key_video_moments_state,
            application_content.speech_transcription_state,
            application_content.behavior_description_state,
            application_content.feature_attribution_state,
            application_content.run_button,
            application_content.clear_button,
            application_content.task_selector,
            application_content.video_input,
            application_content.analysis_busy_overlay,
            application_content.analysis_running_state,
        ],
        queue=True,
        concurrency_limit=1,
        show_progress="hidden",
    )

    key_video_moments_event = analysis_completed_event.then(
        fn=handle_render_key_video_moments,
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

    audio_waveform_event = key_video_moments_event.then(
        fn=handle_create_audio_waveform,
        inputs=[
            application_content.source_video_state,
            language_selector.dropdown,
            application_content.result_cache_state,
        ],
        outputs=audio_waveform_outputs,
        queue=True,
        concurrency_limit=1,
        show_progress="hidden",
    )

    audio_waveform_resize_event = audio_waveform_event.then(
        fn=None,
        inputs=[],
        outputs=[],
        js=AUDIO_WAVEFORM_RESIZE_JS,
        queue=False,
        show_progress="hidden",
    )

    speech_transcription_event = audio_waveform_resize_event.then(
        fn=handle_render_speech_transcription,
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

    behavior_description_event = speech_transcription_event.then(
        fn=handle_render_behavior_description,
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

    feature_attribution_event = behavior_description_event.then(
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

    feature_attribution_resize_event = feature_attribution_event.then(
        fn=None,
        inputs=[],
        outputs=[],
        js=FEATURE_ATTRIBUTION_RESIZE_JS,
        queue=False,
        show_progress="hidden",
    )

    feature_attribution_resize_event.then(
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
