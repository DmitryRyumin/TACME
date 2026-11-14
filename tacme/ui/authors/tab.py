"""
File: tab.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Gradio Authors tab creation.

License: MIT License
"""

import gradio as gr

from tacme.ui.authors.contracts import AuthorsTabComponents
from tacme.ui.authors.rendering import create_authors_content_html


def create_authors_tab(
    language_index: int = 0,
) -> AuthorsTabComponents:
    """Create the localized TACME Authors tab."""

    content = gr.HTML(
        value=create_authors_content_html(
            language_index,
        ),
        elem_id="authors-content",
        elem_classes="authors-content",
    )

    return AuthorsTabComponents(
        content=content,
    )
