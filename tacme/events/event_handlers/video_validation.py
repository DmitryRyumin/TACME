"""
File: video_validation.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Single video-transition coordinator for File, Webcam, Examples and Clear.

License: MIT License
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

import gradio as gr
from gradio.events import Dependency

from tacme.events.application import handle_video_change
from tacme.events.event_handlers.video_filename import (
    VideoSource,
    resolve_video_filename,
)
from tacme.events.event_handlers.video_reset import create_video_reset_updates
from tacme.events.media_info import create_media_content_updates, handle_media_change
from tacme.events.video_state import create_video_component_label
from tacme.localization import get_language_index
from tacme.media.metadata import probe_media_file
from tacme.media.preview import PreviewConversionError, prepare_video_preview
from tacme.media.validation import check_video_container
from tacme.ui.application import ApplicationTabComponents
from tacme.ui.client_scripts import CLEAR_EXAMPLE_SELECTION_JS
from tacme.ui.language_selector import LanguageSelectorComponents


def handle_video_transition(
    video_path: str | None,
    selected_tasks: list[str] | None,
    language: str,
    *,
    source: VideoSource = "file",
) -> tuple[Any, ...]:
    """Validate once and update every video-derived UI state atomically.

    There are no callbacks from ``video_input.change``. In particular,
    resetting rejected video values cannot re-enter this coordinator.
    """

    video_status = check_video_container(video_path)
    video_valid = video_status == "available"
    media_info = handle_media_change(
        video_path if video_valid else None,
        language,
        verified=video_valid,
    )
    preview_path = video_path

    if video_valid and video_path:
        try:
            preview_path = prepare_video_preview(
                video_path,
                media_info[0],
                repair_missing_duration=source == "webcam",
            )
            original_probe = media_info[0]
            if (
                source == "webcam"
                and preview_path != video_path
                and Path(preview_path).suffix.lower() == ".webm"
                and original_probe is not None
                and original_probe.metadata is not None
                and original_probe.metadata.duration_seconds is None
            ):
                playback_probe = probe_media_file(preview_path)
                recovered_duration = (
                    playback_probe.metadata.duration_seconds if playback_probe.metadata is not None else None
                )
                if recovered_duration is None or recovered_duration <= 0:
                    raise PreviewConversionError("Recorded WebM duration was not recovered")
                restored_probe = replace(
                    original_probe,
                    metadata=replace(original_probe.metadata, duration_seconds=recovered_duration),
                )
                media_info = (
                    restored_probe,
                    *media_info[1:4],
                    *create_media_content_updates(restored_probe, get_language_index(language)),
                )
        except PreviewConversionError:
            video_status = "preview_unavailable"
            video_valid = False
            media_info = handle_media_change(None, language)

    filename = resolve_video_filename(
        video_path if video_valid else None,
        language,
        source=source,
    )
    controls = list(
        handle_video_change(
            video_path,
            selected_tasks,
            language,
            video_status=video_status,
            video_filename=filename,
        )
    )
    if video_valid and preview_path != video_path:
        controls[-1] = gr.update(
            value=preview_path,
            label=create_video_component_label(get_language_index(language), filename),
        )

    return (
        *controls,
        video_path if video_valid else None,  # Original file, never the playback copy.
        filename,
        *media_info,
        *create_video_reset_updates(),
    )


def register_video_validation_events(
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
    """Bind explicit user sources; never listen to programmatic ``.change``."""

    outputs = [
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
        application_content.video_input,
        application_content.source_video_state,
        application_content.video_filename_state,
        *media_info_outputs,
        *key_video_moments_outputs,
        *audio_waveform_outputs,
        *speech_transcription_outputs,
        *behavior_description_outputs,
        *feature_attribution_outputs,
        *analysis_summary_outputs,
    ]
    inputs = [
        application_content.video_input,
        application_content.task_selector,
        language_selector.dropdown,
    ]

    def bind_completion(event: Dependency) -> None:
        # Retain the existing upload-checking gate: release it only after
        # validation has set the final status and video value.
        event.then(
            fn=None,
            inputs=[application_content.status, application_content.video_input],
            outputs=[],
            js="(status, videoPath) => { window.__tacmeUploadError?.validated(status, videoPath); return []; }",
            queue=False,
            show_progress="hidden",
        )

    def file_transition(
        video_path: str | None,
        selected_tasks: list[str] | None,
        language: str,
    ) -> tuple[Any, ...]:
        return handle_video_transition(video_path, selected_tasks, language, source="file")

    def webcam_transition(
        video_path: str | None,
        selected_tasks: list[str] | None,
        language: str,
    ) -> tuple[Any, ...]:
        return handle_video_transition(video_path, selected_tasks, language, source="webcam")

    def clear_transition(
        _video_path: str | None,
        selected_tasks: list[str] | None,
        language: str,
    ) -> tuple[Any, ...]:
        # The explicit clear event is authoritative even if Gradio sends an
        # earlier input snapshot while removing the file.
        return handle_video_transition(None, selected_tasks, language, source="clear")

    # Gradio upload/record/clear events describe intentional user actions.
    # Label-only updates and rejected-file resets do not fire these listeners.
    bind_completion(
        application_content.video_input.upload(
            fn=file_transition,
            inputs=inputs,
            outputs=outputs,
            js=CLEAR_EXAMPLE_SELECTION_JS,
            queue=False,
            show_progress="hidden",
        )
    )

    bind_completion(
        application_content.video_input.stop_recording(
            fn=webcam_transition,
            inputs=inputs,
            outputs=outputs,
            js=CLEAR_EXAMPLE_SELECTION_JS,
            queue=False,
            show_progress="hidden",
        )
    )

    bind_completion(
        application_content.video_input.clear(
            fn=clear_transition,
            inputs=inputs,
            outputs=outputs,
            js=CLEAR_EXAMPLE_SELECTION_JS,
            queue=False,
            show_progress="hidden",
        )
    )

    if application_content.examples is not None:
        bind_completion(
            application_content.examples.load_input_event.then(
                fn=file_transition,
                inputs=inputs,
                outputs=outputs,
                queue=False,
                show_progress="hidden",
            )
        )
