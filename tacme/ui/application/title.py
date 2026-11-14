"""
File: title.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Localized application title and version markup.

License: MIT License
"""

from html import escape

from tacme.config import get_config_str
from tacme.localization import get_localized_text


def create_application_title_markdown(
    language_index: int,
) -> str:
    """Create the TACME title with the application version."""

    title_text = get_localized_text(
        "Texts_APP_TITLE",
        language_index,
    ).strip()

    subtitle_text = get_localized_text(
        "Texts_APP_SUBTITLE",
        language_index,
    ).strip()

    subtitle_prefix = f"{title_text}:"

    if subtitle_text.startswith(
        subtitle_prefix,
    ):
        subtitle_text = subtitle_text[len(subtitle_prefix) :].strip()

    app_version = get_config_str(
        "App_VERSION",
        "0.0.0",
    ).strip()

    return (
        f"### {escape(title_text)}: "
        f"{escape(subtitle_text)} "
        '<span class="application-version">'
        f"v{escape(app_version)}"
        "</span>"
    )
