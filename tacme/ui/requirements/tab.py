"""
File: tab.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: TACME Requirements tab implementation.

License: MIT License
"""

import gradio as gr

from tacme.localization import get_localized_text
from tacme.ui.requirements.models import RequirementsTabComponents
from tacme.ui.requirements.rendering import create_initial_requirements_content_html
from tacme.ui.requirements.session import _create_pending_session_state


def create_requirements_tab(
    language_index: int = 0,
) -> RequirementsTabComponents:
    """Create the TACME Requirements tab."""

    title = gr.Markdown(
        (
            "### "
            f"{
                get_localized_text(
                    'Texts_REQUIREMENTS_TITLE',
                    language_index,
                )
            }"
        ),
        elem_classes="requirements-title",
    )

    description = gr.Markdown(
        get_localized_text(
            "Requirements_DESCRIPTION",
            language_index,
        ),
        elem_classes="requirements-description",
    )

    session_state = gr.State(
        value=_create_pending_session_state,
    )

    content = gr.HTML(
        value=create_initial_requirements_content_html(
            language_index,
        ),
        elem_id="requirements-content",
        elem_classes="requirements-content",
    )

    return RequirementsTabComponents(
        title=title,
        description=description,
        content=content,
        session_state=session_state,
    )
