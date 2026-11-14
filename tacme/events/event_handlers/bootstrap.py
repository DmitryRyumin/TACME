"""
File: bootstrap.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Synchronous startup JS and upload-preflight event bindings.

License: MIT License
"""

import gradio as gr

from tacme.events.analysis_summary import ANALYSIS_SUMMARY_JS
from tacme.events.attribution import FEATURE_ATTRIBUTION_JS
from tacme.events.behavior_description import BEHAVIOR_DESCRIPTION_JS
from tacme.events.keyframes import KEY_VIDEO_MOMENTS_JS
from tacme.events.transcription import SPEECH_TRANSCRIPTION_JS
from tacme.ui.client_scripts import EXAMPLES_UI_JS
from tacme.ui.language_selector import LanguageSelectorComponents
from tacme.ui.progress_motion import PROGRESS_MOTION_JS
from tacme.ui.upload_error import UPLOAD_ERROR_UI_JS


def register_client_bootstrap_events(
    gradio_app: gr.Blocks,
    language_selector: LanguageSelectorComponents,
) -> None:
    """Register initial client-side scripts in the original exact order."""

    gradio_app.load(
        fn=None,
        inputs=[
            language_selector.dropdown,
        ],
        outputs=[],
        js=EXAMPLES_UI_JS,
        queue=False,
        show_progress="hidden",
    )

    gradio_app.load(
        fn=None,
        inputs=[],
        outputs=[],
        js=KEY_VIDEO_MOMENTS_JS,
        queue=False,
        show_progress="hidden",
    )

    gradio_app.load(
        fn=None,
        inputs=[],
        outputs=[],
        js=SPEECH_TRANSCRIPTION_JS,
        queue=False,
        show_progress="hidden",
    )

    gradio_app.load(
        fn=None,
        inputs=[],
        outputs=[],
        js=BEHAVIOR_DESCRIPTION_JS,
        queue=False,
        show_progress="hidden",
    )

    gradio_app.load(
        fn=None,
        inputs=[],
        outputs=[],
        js=FEATURE_ATTRIBUTION_JS,
        queue=False,
        show_progress="hidden",
    )

    gradio_app.load(
        fn=None,
        inputs=[],
        outputs=[],
        js=ANALYSIS_SUMMARY_JS,
        queue=False,
        show_progress="hidden",
    )

    gradio_app.load(
        fn=None,
        inputs=[],
        outputs=[],
        js=PROGRESS_MOTION_JS,
        queue=False,
        show_progress="hidden",
    )

    # Observe errors before Gradio delivers a video to Python. The controller
    # only reads native file readability and Gradio's visible upload messages.
    gradio_app.load(
        fn=None,
        inputs=[language_selector.dropdown],
        outputs=[],
        js=UPLOAD_ERROR_UI_JS,
        queue=False,
        show_progress="hidden",
    )

    language_selector.dropdown.change(
        fn=None,
        inputs=[language_selector.dropdown],
        outputs=[],
        js=UPLOAD_ERROR_UI_JS,
        queue=False,
        show_progress="hidden",
    )
