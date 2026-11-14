"""
File: requirements.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Requirements tab Gradio event registration.

License: MIT License
"""

import gradio as gr

from tacme.ui.language_selector import LanguageSelectorComponents
from tacme.ui.requirements import (
    RequirementsTabComponents,
    render_requirements_session_state,
    stream_requirements_session_state,
)


def register_requirements_events(
    gradio_app: gr.Blocks,
    language_selector: LanguageSelectorComponents,
    requirements_content: RequirementsTabComponents,
) -> None:
    """Register Requirements events in their original order."""

    gradio_app.load(
        fn=stream_requirements_session_state,
        inputs=[
            requirements_content.session_state,
        ],
        outputs=[
            requirements_content.session_state,
        ],
        queue=True,
        concurrency_limit=4,
        show_progress="hidden",
    )

    requirements_content.session_state.change(
        fn=render_requirements_session_state,
        inputs=[
            language_selector.dropdown,
            requirements_content.session_state,
        ],
        outputs=[
            requirements_content.content,
        ],
        queue=False,
        show_progress="hidden",
    )
