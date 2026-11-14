"""
File: compatibility.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: TACME media compatibility HTML and state marker.

License: MIT License
"""

from html import escape

from tacme.localization import get_localized_text
from tacme.media.formatting.components import create_media_unavailable_html
from tacme.media.metadata import MediaProbeResult
from tacme.media.validation import validate_media_for_tacme


def _create_compatibility_tab_state_marker(
    probe_result: MediaProbeResult | None,
) -> str:
    """Create an always-available compatibility state marker for the tab icon."""

    is_ready = (
        probe_result is not None
        and validate_media_for_tacme(
            probe_result,
        ).is_ready
    )

    state_class = "is-ready" if is_ready else "is-not-ready"

    return f'<span class="media-compatibility-tab-state {state_class}" aria-hidden="true"></span>'


def create_media_compatibility_html(
    probe_result: MediaProbeResult | None,
    language_index: int,
) -> str:
    """Create the TACME compatibility tab content."""

    if probe_result is None:
        return create_media_unavailable_html(
            language_index,
        )

    compatibility = validate_media_for_tacme(
        probe_result,
    )

    check_rows = "".join(
        (
            f"""
<div class="media-compatibility-check {"is-passed" if check.passed else "is-failed"}">
    <span class="media-compatibility-marker">{"&#10003;" if check.passed else "&#10005;"}</span>
    <span>{escape(get_localized_text(f"Labels_MEDIA_CHECK_{check.key}", language_index))}</span>
</div>
""".strip()
        )
        for check in compatibility.checks
    )

    if compatibility.is_ready:
        status_class = "is-ready"
        status_title = get_localized_text(
            "Texts_MEDIA_COMPATIBILITY_READY_TITLE",
            language_index,
        )
        status_text = get_localized_text(
            "Texts_MEDIA_COMPATIBILITY_READY",
            language_index,
        )
    else:
        status_class = "is-not-ready"
        status_title = get_localized_text(
            "Texts_MEDIA_COMPATIBILITY_NOT_READY_TITLE",
            language_index,
        )
        status_text = get_localized_text(
            "Texts_MEDIA_COMPATIBILITY_NOT_READY",
            language_index,
        )

    return f"""
<section class="media-compatibility-content">
    <div class="media-compatibility-status {status_class}">
        <strong>{escape(status_title)}</strong>
        <span>{escape(status_text)}</span>
    </div>
    <div class="media-compatibility-checks">{check_rows}</div>
</section>
""".strip()
