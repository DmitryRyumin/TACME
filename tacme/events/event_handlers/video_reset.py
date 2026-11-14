"""
File: video_reset.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Gradio bindings for resetting derived results after a video change.

License: MIT License
"""

from typing import Any

from tacme.events.analysis_summary import handle_clear_analysis_summary
from tacme.events.attribution import handle_clear_feature_attribution
from tacme.events.behavior_description import handle_clear_behavior_description
from tacme.events.keyframes import handle_clear_key_video_moments
from tacme.events.transcription import handle_clear_speech_transcription
from tacme.events.waveform import handle_clear_audio_waveform


def create_video_reset_updates() -> tuple[Any, ...]:
    """Clear derived sections as part of one video transition.

    This helper does not register any Gradio listeners; the coordinator owns
    all video-input events so resetting results cannot trigger a validation.
    """

    return (
        *handle_clear_key_video_moments(),
        *handle_clear_audio_waveform(),
        *handle_clear_speech_transcription(),
        *handle_clear_behavior_description(),
        *handle_clear_feature_attribution(),
        *handle_clear_analysis_summary(),
    )
