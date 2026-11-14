"""
File: components.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Shared HTML components for TACME media information.

License: MIT License
"""

from html import escape

from tacme.localization import get_localized_text


def _create_info_row(
    label: str,
    value: str,
    *,
    code: bool = False,
) -> str:
    """Create one compact metadata row."""

    safe_label = escape(
        label,
    )
    safe_value = escape(
        value,
    )
    value_html = f"<code>{safe_value}</code>" if code else safe_value

    return f"""
<div class="media-info-row">
    <div class="media-info-label">{safe_label}</div>
    <div class="media-info-value">{value_html}</div>
</div>
""".strip()


def _create_summary_card(
    label: str,
    value: str,
) -> str:
    """Create one overview summary card."""

    return f"""
<div class="media-summary-card">
    <div class="media-summary-label">{escape(label)}</div>
    <div class="media-summary-value">{escape(value)}</div>
</div>
""".strip()


def create_media_unavailable_html(
    language_index: int,
) -> str:
    """Create a localized unavailable-media message."""

    title = escape(
        get_localized_text(
            "Labels_MEDIA_INFO_UNAVAILABLE_TITLE",
            language_index,
        )
    )
    message = escape(
        get_localized_text(
            "Texts_MEDIA_INFO_UNAVAILABLE",
            language_index,
        )
    )

    return f"""
<section class="media-info-empty-state">
    <h4>{title}</h4>
    <p>{message}</p>
</section>
""".strip()
