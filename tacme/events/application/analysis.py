"""
File: analysis.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Original Gradio streaming analysis handlers and progress instrumentation.

License: MIT License
"""

from collections.abc import Iterator, Sequence
from queue import Empty, Queue
from threading import RLock, Thread
from time import perf_counter, sleep
from typing import Any, cast

import gradio as gr

from tacme.events.application.results import create_result_updates
from tacme.events.application.status import (
    _video_file_status_text,
    create_analysis_completed_status,
)
from tacme.inference import TacmeInferenceRun, get_tacme_inference_engine
from tacme.localization import get_language_index, get_localized_text
from tacme.media.metadata import MediaProbeResult
from tacme.media.validation import VideoFileStatus, check_video_container
from tacme.performance import (
    PROGRESS_STAGE_KEYS,
    StageProgressEvent,
    StageTiming,
    create_analysis_timing,
)
from tacme.ui.progress_modal import create_progress_modal
from tacme.visualization.keyframes import KeyVideoMoments, create_key_video_moments

_PREDICTION_LOCK = RLock()


def _create_busy_overlay(language_index: int) -> str:
    """Show a modal with actual stage progress, initially at zero."""

    return create_progress_modal(language_index, {}, {}, 0.0)


def _progress_overlay_only(value: str) -> tuple[Any, ...]:
    """Stream just the overlay, preserving all other 16 Gradio outputs."""

    return (*[gr.update() for _ in range(15)], value, gr.update())


def handle_analysis_started(
    video_path: str | None,
    language: str,
) -> tuple[Any, ...]:
    """Lock TACME controls only when the input file is still accessible."""

    language_index = get_language_index(language)
    video_status = check_video_container(video_path)
    if video_status != "available":
        return (
            gr.update(interactive=False),
            gr.update(interactive=bool(video_path)),
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(
                value=_video_file_status_text(video_status, language_index),
            ),
            "",
            False,
            gr.update(visible=False),
        )

    return (
        gr.update(
            interactive=False,
        ),
        gr.update(
            interactive=False,
        ),
        gr.update(
            interactive=False,
        ),
        gr.update(
            interactive=False,
        ),
        gr.update(
            value=get_localized_text(
                "Texts_STATUS_ANALYSIS_RUNNING",
                language_index,
            ),
        ),
        _create_busy_overlay(
            language_index,
        ),
        True,
        gr.update(visible=False),
    )


def handle_run_analysis(
    video_path: str | None,
    selected_tasks: list[str] | None,
    language: str,
    media_info_state: MediaProbeResult | None,
) -> Iterator[tuple[Any, ...]]:
    """Stream real stage measurements while running inference in a worker thread."""

    language_index = get_language_index(
        language,
    )

    video_status = check_video_container(video_path)
    if video_status != "available":
        yield (
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
            None,
            None,
            None,
            None,
            None,
            None,
            gr.update(
                interactive=False,
            ),
            gr.update(
                interactive=bool(video_path),
            ),
            gr.update(
                interactive=False,
            ),
            gr.update(
                interactive=True,
            ),
            "",
            False,
        )
        return

    if not selected_tasks:
        yield (
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
            None,
            None,
            None,
            None,
            None,
            None,
            gr.update(
                interactive=False,
            ),
            gr.update(
                interactive=True,
            ),
            gr.update(
                interactive=True,
            ),
            gr.update(
                interactive=True,
            ),
            "",
            False,
        )
        return

    validated_video_path = cast(str, video_path)

    duration_seconds = (
        media_info_state.metadata.duration_seconds
        if media_info_state is not None and media_info_state.metadata is not None
        else None
    )

    events: Queue[tuple[str, Any]] = Queue()
    display_started_at = perf_counter()
    stage_states: dict[str, str] = {}
    stage_durations: dict[str, float] = {}
    active_started_at: float | None = None

    def publish_progress(event: StageProgressEvent) -> None:
        """Send instrumentation updates from the model thread to Gradio."""

        events.put(("progress", event))

    def inference_worker() -> None:
        """Run the shared inference engine without blocking streamed UI updates."""

        try:
            with _PREDICTION_LOCK:
                file_status = check_video_container(validated_video_path)
                if file_status != "available":
                    events.put(("file_error", file_status))
                    return
                engine = get_tacme_inference_engine()
                started_at = perf_counter()
                inference_run = engine.predict_with_artifacts(
                    validated_video_path,
                    progress_callback=publish_progress,
                )
                elapsed = perf_counter() - started_at
            events.put(("success", (inference_run, elapsed, str(engine.device))))
        except Exception as error:
            # A local input may disappear between validation and decoder startup.
            file_status = check_video_container(validated_video_path)
            events.put(("file_error", file_status) if file_status != "available" else ("error", error))

    worker = Thread(target=inference_worker, daemon=True, name="tacme-analysis")
    worker.start()

    finished_run: tuple[TacmeInferenceRun, float, str] | None = None
    while finished_run is None:
        try:
            event_type, event_value = events.get(timeout=1.0)
        except Empty:
            yield _progress_overlay_only(
                create_progress_modal(
                    language_index,
                    stage_states,
                    stage_durations,
                    perf_counter() - display_started_at,
                    active_elapsed_seconds=(
                        perf_counter() - active_started_at if active_started_at is not None else 0.0
                    ),
                )
            )
            continue

        if event_type == "progress":
            progress = cast(StageProgressEvent, event_value)
            stage_states[progress.stage] = progress.status
            if progress.status == "started":
                active_started_at = perf_counter()
            elif progress.status in ("completed", "failed"):
                active_started_at = None
            if progress.elapsed_seconds is not None:
                stage_durations[progress.stage] = progress.elapsed_seconds
            yield _progress_overlay_only(
                create_progress_modal(
                    language_index,
                    stage_states,
                    stage_durations,
                    perf_counter() - display_started_at,
                    active_elapsed_seconds=(
                        perf_counter() - active_started_at if active_started_at is not None else 0.0
                    ),
                )
            )
        elif event_type == "success":
            finished_run = cast(tuple[TacmeInferenceRun, float, str], event_value)
        else:
            if event_type == "file_error":
                failure_text = _video_file_status_text(
                    cast(VideoFileStatus, event_value),
                    language_index,
                )
            else:
                error = cast(Exception, event_value)
                failure_text = f"{get_localized_text('Texts_STATUS_ANALYSIS_FAILED', language_index)} {error}"

            yield (
                gr.update(
                    value=failure_text,
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
                None,
                None,
                None,
                None,
                None,
                None,
                gr.update(
                    interactive=event_type != "file_error",
                ),
                gr.update(
                    interactive=True,
                ),
                gr.update(
                    interactive=event_type != "file_error",
                ),
                gr.update(
                    interactive=True,
                ),
                "",
                False,
            )
            return

    # Display a brief, truthful 12/12 completion state before removing the overlay.
    yield _progress_overlay_only(
        create_progress_modal(
            language_index,
            stage_states,
            stage_durations,
            perf_counter() - display_started_at,
            finished=True,
        )
    )
    sleep(0.55)  # Presentation only; deliberately excluded from inference RTF.

    inference_run, analysis_seconds, inference_device = finished_run
    result = inference_run.result

    analysis_timing = create_analysis_timing(
        analysis_seconds,
        duration_seconds,
        device=inference_device,
        stage_timings=tuple(
            StageTiming(stage=stage, elapsed_seconds=stage_durations[stage])
            for stage in cast(Sequence[str], PROGRESS_STAGE_KEYS)
            if stage_states.get(stage) == "completed" and stage in stage_durations
        ),
    )

    key_video_moments: KeyVideoMoments | None

    try:
        key_video_moments = create_key_video_moments(
            inference_run.scene_artifacts,
            duration_seconds=duration_seconds,
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
    ):
        key_video_moments = None

    (
        results_container,
        emotion_result,
        personality_result,
        ambivalence_result,
    ) = create_result_updates(
        result,
        selected_tasks,
        language_index,
    )

    yield (
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
        result,
        analysis_timing,
        key_video_moments,
        inference_run.transcription,
        inference_run.behavior_description,
        inference_run.feature_attribution,
        gr.update(
            interactive=True,
        ),
        gr.update(
            interactive=True,
        ),
        gr.update(
            interactive=True,
        ),
        gr.update(
            interactive=True,
        ),
        "",
        False,
    )
