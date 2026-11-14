"""
File: tab.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Original Gradio application-tab component factory, with its body unchanged.

License: MIT License
"""

from typing import Final

import gradio as gr
from gradio.helpers import Examples as GradioExamples

from tacme.localization import TASK_KEYS, get_localized_text, get_task_choices
from tacme.ui.application.contracts import ApplicationTabComponents
from tacme.ui.application.examples import (
    get_example_video_labels,
    get_existing_example_video_paths,
)
from tacme.ui.application.title import create_application_title_markdown

EXAMPLES_PER_PAGE: Final = 6


def create_application_tab(
    language_index: int = 0,
) -> ApplicationTabComponents:
    """Create the main TACME application tab."""

    task_choices = get_task_choices(language_index)

    title = gr.Markdown(
        value=create_application_title_markdown(
            language_index,
        ),
        elem_classes="application-compact-title",
    )

    video_filename_state = gr.State(
        value=None,
    )

    media_info_state = gr.State(
        value=None,
    )

    # Keep the unmodified upload for inference; gr.Video may show a preview.
    source_video_state = gr.State(
        value=None,
    )

    key_video_moments_state = gr.State(
        value=None,
    )

    audio_waveform_state = gr.State(
        value=None,
    )

    speech_transcription_state = gr.State(
        value=None,
    )

    behavior_description_state = gr.State(
        value=None,
    )

    feature_attribution_state = gr.State(
        value=None,
    )

    analysis_running_state = gr.State(
        value=False,
    )

    analysis_timing_state = gr.State(
        value=None,
    )

    result_cache_state = gr.State(
        value=None,
    )

    analysis_busy_overlay = gr.HTML(
        value="",
        visible=True,
        elem_classes="application-floating-html",
    )

    with gr.Row(
        elem_classes="application-input-row",
    ):
        with gr.Column(
            scale=5,
            min_width=420,
            elem_classes="application-video-column",
        ):
            video_input = gr.Video(
                label=get_localized_text(
                    "Labels_VIDEO_INPUT",
                    language_index,
                ),
                sources=[
                    "upload",
                    "webcam",
                ],
                format=None,  # Validate the source before creating a browser-compatible preview.
                interactive=True,
                buttons=[
                    "download",
                ],
                elem_id="application-video-input",
                elem_classes="application-video-input",
            )

        with gr.Column(
            scale=3,
            min_width=260,
            elem_classes="application-examples-column",
        ):
            example_video_paths = get_existing_example_video_paths()

            examples_title = gr.Markdown(
                value=(
                    "### "
                    + get_localized_text(
                        "Labels_EXAMPLES",
                        language_index,
                    )
                ),
                elem_classes="application-examples-title",
            )

            examples: GradioExamples | None = None

            if example_video_paths:
                examples = gr.Examples(
                    examples=example_video_paths,
                    inputs=[
                        video_input,
                    ],
                    cache_examples=False,
                    examples_per_page=EXAMPLES_PER_PAGE,
                    label="",
                    example_labels=(
                        get_example_video_labels(
                            example_video_paths,
                        )
                    ),
                    elem_id="application-examples",
                )

                examples_placeholder = gr.Markdown(
                    value="",
                    visible=False,
                    elem_classes=("application-examples-placeholder"),
                )

            else:
                examples_placeholder = gr.Markdown(
                    value=get_localized_text(
                        "Texts_EXAMPLES_PLACEHOLDER",
                        language_index,
                    ),
                    visible=True,
                    elem_classes=("application-examples-placeholder"),
                )

        with gr.Column(
            scale=4,
            min_width=300,
            elem_classes="application-task-column",
        ):
            task_selector = gr.CheckboxGroup(
                choices=task_choices,
                value=list(TASK_KEYS),
                label=get_localized_text(
                    "Labels_TASKS",
                    language_index,
                ),
                visible=False,
                interactive=True,
                elem_classes="application-task-selector",
            )

    with gr.Row(
        elem_classes="application-action-row",
    ):
        run_button = gr.Button(
            value=get_localized_text(
                "Labels_RUN_ANALYSIS",
                language_index,
            ),
            variant="primary",
            size="lg",
            interactive=False,
            elem_id="run-button",
            elem_classes="application-run-button",
        )

        clear_button = gr.Button(
            value=get_localized_text(
                "Labels_CLEAR",
                language_index,
            ),
            variant="secondary",
            size="lg",
            interactive=False,
            elem_id="clear-button",
            elem_classes="application-clear-button",
        )

    with gr.Row(
        elem_classes="application-status-row",
    ):
        with (
            gr.Column(
                scale=7,
                min_width=560,
                elem_classes="application-status-column",
            ),
            gr.Row(elem_classes="application-status-inline"),
        ):
            status = gr.Markdown(
                value=get_localized_text(
                    "Texts_STATUS_READY",
                    language_index,
                ),
                elem_classes="application-status",
            )
            processing_details_button = gr.Button(
                value=get_localized_text("Texts_DETAILS_BUTTON", language_index),
                variant="secondary",
                size="sm",
                visible=False,
                elem_id="application-processing-details-button",
                elem_classes="application-processing-details-button",
            )

        with gr.Column(
            scale=3,
            min_width=300,
            visible=False,
            elem_id="media-info-button-column",
            elem_classes="application-media-info-button-column",
        ) as media_info_button_column:
            media_info_button = gr.Button(
                value=get_localized_text(
                    "Labels_MEDIA_INFO_TITLE",
                    language_index,
                ),
                variant="secondary",
                size="lg",
                interactive=False,
                visible=True,
                elem_id="media-info-button",
                elem_classes="application-media-info-button",
            )

    with (
        gr.Column(
            visible=False,
            elem_id="media-info-modal",
            elem_classes="media-info-modal-backdrop",
        ) as media_info_modal,
        gr.Column(
            elem_classes="media-info-modal-card",
        ),
    ):
        with gr.Row(
            elem_classes="media-info-modal-header",
        ):
            with gr.Column(
                scale=1,
                min_width=0,
                elem_classes="media-info-modal-title-column",
            ):
                media_info_modal_title = gr.Markdown(
                    value=(
                        "### "
                        + get_localized_text(
                            "Labels_MEDIA_INFO_TITLE",
                            language_index,
                        )
                    ),
                    elem_classes="media-info-modal-title",
                )

            with gr.Column(
                scale=0,
                min_width=42,
                elem_classes="media-info-modal-close-column",
            ):
                media_info_modal_close_button = gr.Button(
                    value="Close",
                    variant="secondary",
                    size="sm",
                    elem_id="media-info-modal-close-button",
                    elem_classes="media-info-modal-close-button",
                )

        with gr.Tabs(
            selected="overview",
            elem_id="media-info-tabs",
            elem_classes="media-info-tabs",
        ) as media_info_tabs:
            with gr.Tab(
                label=get_localized_text(
                    "Labels_MEDIA_TAB_OVERVIEW",
                    language_index,
                ),
                id="overview",
                elem_classes="media-info-tab",
            ) as media_overview_tab:
                media_overview_content = gr.HTML(
                    value="",
                    elem_classes="media-info-tab-content",
                )

            with gr.Tab(
                label=get_localized_text(
                    "Labels_MEDIA_TAB_VIDEO",
                    language_index,
                ),
                id="video",
                elem_classes="media-info-tab",
            ) as media_video_tab:
                media_video_content = gr.HTML(
                    value="",
                    elem_classes="media-info-tab-content",
                )

            with gr.Tab(
                label=get_localized_text(
                    "Labels_MEDIA_TAB_AUDIO",
                    language_index,
                ),
                id="audio",
                elem_classes="media-info-tab",
            ) as media_audio_tab:
                media_audio_content = gr.HTML(
                    value="",
                    elem_classes="media-info-tab-content",
                )

            with gr.Tab(
                label=get_localized_text(
                    "Labels_MEDIA_TAB_COMPATIBILITY",
                    language_index,
                ),
                id="compatibility",
                elem_classes="media-info-tab",
            ) as media_compatibility_tab:
                media_compatibility_content = gr.HTML(
                    value="",
                    elem_classes="media-info-tab-content",
                )

    with gr.Column(
        visible=False,
        elem_id="application-key-video-moments-container",
        elem_classes="application-key-video-moments-container",
    ) as key_video_moments_container:
        key_video_moments_title = gr.Markdown(
            value=(
                "### "
                + get_localized_text(
                    "Labels_KEY_VIDEO_MOMENTS",
                    language_index,
                )
            ),
            elem_classes="application-key-video-moments-title",
        )

        key_video_moments_content = gr.HTML(
            value="",
            elem_id="application-key-video-moments-content",
            elem_classes="application-key-video-moments-html",
        )

    with gr.Row(
        elem_classes="application-audio-analysis-row",
    ):
        with gr.Column(
            scale=7,
            min_width=0,
            visible=False,
            elem_id="application-audio-waveform-container",
            elem_classes="application-audio-waveform-container",
        ) as audio_waveform_container:
            audio_waveform_title = gr.Markdown(
                value=(
                    "### "
                    + get_localized_text(
                        "Labels_AUDIO_WAVEFORM",
                        language_index,
                    )
                ),
                elem_classes="application-audio-waveform-title",
            )

            audio_waveform_plot = gr.Plot(
                value=None,
                show_label=False,
                container=False,
                elem_id="application-audio-waveform-plot",
                elem_classes="application-audio-waveform-plot",
            )

        with gr.Column(
            scale=3,
            min_width=320,
            visible=False,
            elem_id="application-speech-transcription-container",
            elem_classes="application-speech-transcription-container",
        ) as speech_transcription_container:
            speech_transcription_title = gr.Markdown(
                value=(
                    "### "
                    + get_localized_text(
                        "Labels_SPEECH_TRANSCRIPTION",
                        language_index,
                    )
                ),
                elem_classes="application-speech-transcription-title",
            )

            speech_transcription_content = gr.HTML(
                value="",
                elem_id="application-speech-transcription-content",
                elem_classes="application-speech-transcription-html",
            )

    with gr.Column(
        visible=False,
        elem_id="application-behavior-description-container",
        elem_classes="application-behavior-description-container",
    ) as behavior_description_container:
        behavior_description_title = gr.Markdown(
            value=(
                "### "
                + get_localized_text(
                    "Labels_OBSERVED_BEHAVIOR",
                    language_index,
                )
            ),
            elem_classes="application-behavior-description-title",
        )

        behavior_description_content = gr.HTML(
            value="",
            elem_id="application-behavior-description-content",
            elem_classes="application-behavior-description-html",
        )

    with gr.Column(
        visible=False,
        elem_classes="application-results-container",
    ) as results_container:
        predictions_title = gr.Markdown(
            value=(
                "### "
                + get_localized_text(
                    "Labels_PREDICTIONS",
                    language_index,
                )
            ),
            elem_classes="application-predictions-title",
        )

        with gr.Row(
            elem_classes="application-results-row",
        ):
            emotion_result = gr.HTML(
                value="",
                visible=True,
                elem_classes=[
                    "application-result-card",
                    "application-emotion-result",
                ],
            )

            personality_result = gr.HTML(
                value="",
                visible=True,
                elem_classes=[
                    "application-result-card",
                    "application-personality-result",
                ],
            )

            ambivalence_result = gr.HTML(
                value="",
                visible=True,
                elem_classes=[
                    "application-result-card",
                    "application-ambivalence-result",
                ],
            )

    with gr.Column(
        visible=False,
        elem_id="application-feature-attribution-container",
        elem_classes="application-feature-attribution-container",
    ) as feature_attribution_container:
        feature_attribution_title = gr.Markdown(
            value=(
                "### "
                + get_localized_text(
                    "Labels_ATTRIBUTION_SECTION_TITLE",
                    language_index,
                )
            ),
            elem_classes="application-feature-attribution-title",
        )

        with gr.Row(
            elem_classes="application-feature-attribution-controls",
        ):
            with gr.Column(
                scale=42,
                min_width=0,
                elem_classes=[
                    "application-feature-attribution-control-group",
                    "application-feature-attribution-task-group",
                ],
            ):
                feature_attribution_task = gr.Dropdown(
                    choices=[],
                    value=None,
                    type="value",
                    filterable=True,
                    allow_custom_value=False,
                    container=True,
                    show_label=True,
                    label=get_localized_text(
                        "Labels_ATTRIBUTION_TASK",
                        language_index,
                    ),
                    interactive=True,
                    elem_id="application-feature-attribution-task",
                    elem_classes="application-feature-attribution-control",
                )

            with gr.Column(
                scale=34,
                min_width=0,
                elem_classes=[
                    "application-feature-attribution-control-group",
                    "application-feature-attribution-target-group",
                ],
            ):
                feature_attribution_target = gr.Dropdown(
                    choices=[],
                    value=None,
                    type="value",
                    filterable=True,
                    allow_custom_value=False,
                    container=True,
                    show_label=True,
                    label=get_localized_text(
                        "Labels_ATTRIBUTION_TARGET",
                        language_index,
                    ),
                    interactive=True,
                    elem_id="application-feature-attribution-target",
                    elem_classes="application-feature-attribution-control",
                )

            with gr.Column(
                scale=24,
                min_width=0,
                elem_classes=[
                    "application-feature-attribution-control-group",
                    "application-feature-attribution-view-group",
                ],
            ):
                feature_attribution_view = gr.Dropdown(
                    choices=[],
                    value=None,
                    type="value",
                    filterable=True,
                    allow_custom_value=False,
                    container=True,
                    show_label=True,
                    label=get_localized_text(
                        "Labels_ATTRIBUTION_VIEW",
                        language_index,
                    ),
                    interactive=True,
                    elem_id="application-feature-attribution-view",
                    elem_classes="application-feature-attribution-control",
                )

        with gr.Row(
            elem_classes="application-feature-attribution-body",
        ):
            with gr.Column(
                scale=4,
                min_width=0,
                elem_classes="application-feature-attribution-plot-column",
            ):
                feature_attribution_plot = gr.Plot(
                    value=None,
                    show_label=False,
                    container=False,
                    elem_id="application-feature-attribution-plot",
                    elem_classes="application-feature-attribution-plot",
                )

            with gr.Column(
                scale=1,
                min_width=250,
                elem_classes="application-feature-attribution-summary-column",
            ):
                feature_attribution_summary = gr.HTML(
                    value="",
                    elem_id="application-feature-attribution-summary",
                    elem_classes="application-feature-attribution-summary-html",
                )

        feature_attribution_note = gr.Markdown(
            value="",
            visible=False,
            elem_classes="application-feature-attribution-note",
        )

    with gr.Column(
        visible=False,
        elem_id="application-analysis-summary-container",
        elem_classes="application-analysis-summary-container",
    ) as analysis_summary_container:
        analysis_summary_content = gr.HTML(
            value="",
            elem_id="application-analysis-summary-content",
            elem_classes="application-analysis-summary-html",
        )

    with (
        gr.Column(
            visible=False,
            elem_id="application-processing-details-modal",
            elem_classes="application-processing-details-backdrop",
        ) as processing_details_modal,
        gr.Column(elem_classes="application-processing-details-card"),
    ):
        with gr.Row(elem_classes="application-processing-details-header"):
            processing_details_title = gr.Markdown(
                value="### " + get_localized_text("Texts_DETAILS_TITLE", language_index),
                elem_classes="application-processing-details-title",
            )
            processing_details_close_button = gr.Button(
                value="Close",
                variant="secondary",
                size="sm",
                elem_id="application-processing-details-close",
                elem_classes="application-processing-details-close",
            )
        processing_details_content = gr.HTML(
            value="",
            elem_classes="application-processing-details-content",
        )

    return ApplicationTabComponents(
        title=title,
        video_input=video_input,
        source_video_state=source_video_state,
        video_filename_state=video_filename_state,
        media_info_state=media_info_state,
        examples=examples,
        examples_title=examples_title,
        examples_placeholder=examples_placeholder,
        task_selector=task_selector,
        run_button=run_button,
        clear_button=clear_button,
        status=status,
        processing_details_button=processing_details_button,
        processing_details_modal=processing_details_modal,
        processing_details_title=processing_details_title,
        processing_details_close_button=processing_details_close_button,
        processing_details_content=processing_details_content,
        media_info_button_column=media_info_button_column,
        media_info_button=media_info_button,
        media_info_modal=media_info_modal,
        media_info_modal_title=media_info_modal_title,
        media_info_modal_close_button=media_info_modal_close_button,
        media_info_tabs=media_info_tabs,
        media_overview_tab=media_overview_tab,
        media_video_tab=media_video_tab,
        media_audio_tab=media_audio_tab,
        media_compatibility_tab=media_compatibility_tab,
        media_overview_content=media_overview_content,
        media_video_content=media_video_content,
        media_audio_content=media_audio_content,
        media_compatibility_content=media_compatibility_content,
        key_video_moments_state=key_video_moments_state,
        key_video_moments_container=key_video_moments_container,
        key_video_moments_title=key_video_moments_title,
        key_video_moments_content=key_video_moments_content,
        audio_waveform_state=audio_waveform_state,
        audio_waveform_container=audio_waveform_container,
        audio_waveform_title=audio_waveform_title,
        audio_waveform_plot=audio_waveform_plot,
        speech_transcription_state=speech_transcription_state,
        speech_transcription_container=speech_transcription_container,
        speech_transcription_title=speech_transcription_title,
        speech_transcription_content=speech_transcription_content,
        behavior_description_state=behavior_description_state,
        behavior_description_container=behavior_description_container,
        behavior_description_title=behavior_description_title,
        behavior_description_content=behavior_description_content,
        results_container=results_container,
        predictions_title=predictions_title,
        emotion_result=emotion_result,
        personality_result=personality_result,
        ambivalence_result=ambivalence_result,
        feature_attribution_state=feature_attribution_state,
        feature_attribution_container=feature_attribution_container,
        feature_attribution_title=feature_attribution_title,
        feature_attribution_task=feature_attribution_task,
        feature_attribution_target=feature_attribution_target,
        feature_attribution_view=feature_attribution_view,
        feature_attribution_plot=feature_attribution_plot,
        feature_attribution_summary=feature_attribution_summary,
        feature_attribution_note=feature_attribution_note,
        analysis_summary_container=analysis_summary_container,
        analysis_summary_content=analysis_summary_content,
        analysis_running_state=analysis_running_state,
        analysis_timing_state=analysis_timing_state,
        result_cache_state=result_cache_state,
        analysis_busy_overlay=analysis_busy_overlay,
    )
