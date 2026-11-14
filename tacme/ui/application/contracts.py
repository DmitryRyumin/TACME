"""
File: contracts.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Data contract shared by the main application tab and its event handlers.

License: MIT License
"""

from dataclasses import dataclass

import gradio as gr
from gradio.helpers import Examples as GradioExamples


@dataclass(
    frozen=True,
    slots=True,
)
class ApplicationTabComponents:
    """Components created inside the main TACME application tab."""

    title: gr.Markdown

    video_input: gr.Video
    source_video_state: gr.State
    video_filename_state: gr.State
    media_info_state: gr.State

    examples: GradioExamples | None
    examples_title: gr.Markdown
    examples_placeholder: gr.Markdown

    task_selector: gr.CheckboxGroup

    run_button: gr.Button
    clear_button: gr.Button

    status: gr.Markdown
    processing_details_button: gr.Button
    processing_details_modal: gr.Column
    processing_details_title: gr.Markdown
    processing_details_close_button: gr.Button
    processing_details_content: gr.HTML

    media_info_button_column: gr.Column
    media_info_button: gr.Button

    media_info_modal: gr.Column
    media_info_modal_title: gr.Markdown
    media_info_modal_close_button: gr.Button
    media_info_tabs: gr.Tabs

    media_overview_tab: gr.Tab
    media_video_tab: gr.Tab
    media_audio_tab: gr.Tab
    media_compatibility_tab: gr.Tab

    media_overview_content: gr.HTML
    media_video_content: gr.HTML
    media_audio_content: gr.HTML
    media_compatibility_content: gr.HTML

    key_video_moments_state: gr.State
    key_video_moments_container: gr.Column
    key_video_moments_title: gr.Markdown
    key_video_moments_content: gr.HTML

    audio_waveform_state: gr.State
    audio_waveform_container: gr.Column
    audio_waveform_title: gr.Markdown
    audio_waveform_plot: gr.Plot

    speech_transcription_state: gr.State
    speech_transcription_container: gr.Column
    speech_transcription_title: gr.Markdown
    speech_transcription_content: gr.HTML

    behavior_description_state: gr.State
    behavior_description_container: gr.Column
    behavior_description_title: gr.Markdown
    behavior_description_content: gr.HTML

    results_container: gr.Column
    predictions_title: gr.Markdown
    emotion_result: gr.HTML
    personality_result: gr.HTML
    ambivalence_result: gr.HTML

    feature_attribution_state: gr.State
    feature_attribution_container: gr.Column
    feature_attribution_title: gr.Markdown
    feature_attribution_task: gr.Dropdown
    feature_attribution_target: gr.Dropdown
    feature_attribution_view: gr.Dropdown
    feature_attribution_plot: gr.Plot
    feature_attribution_summary: gr.HTML
    feature_attribution_note: gr.Markdown

    analysis_summary_container: gr.Column
    analysis_summary_content: gr.HTML

    analysis_running_state: gr.State
    analysis_timing_state: gr.State
    result_cache_state: gr.State

    analysis_busy_overlay: gr.HTML
