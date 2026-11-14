"""
File: overview.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Overview-tab HTML for TACME media information.

License: MIT License
"""

from tacme.localization import get_localized_text
from tacme.media.formatting.compatibility import _create_compatibility_tab_state_marker
from tacme.media.formatting.components import (
    _create_info_row,
    _create_summary_card,
    create_media_unavailable_html,
)
from tacme.media.formatting.values import (
    _format_audio_summary,
    _format_bit_rate,
    _format_container,
    _format_duration,
    _format_file_size,
    _format_frame_rate,
    _format_resolution,
)
from tacme.media.metadata import MediaProbeResult


def create_media_overview_html(
    probe_result: MediaProbeResult | None,
    language_index: int,
) -> str:
    """Create the Overview tab content."""

    compatibility_state_marker = _create_compatibility_tab_state_marker(
        probe_result,
    )

    if probe_result is None or probe_result.metadata is None:
        return compatibility_state_marker + create_media_unavailable_html(
            language_index,
        )

    metadata = probe_result.metadata

    summary_cards = "".join(
        (
            _create_summary_card(
                get_localized_text(
                    "Labels_MEDIA_INFO_DURATION",
                    language_index,
                ),
                _format_duration(
                    metadata.duration_seconds,
                    language_index,
                ),
            ),
            _create_summary_card(
                get_localized_text(
                    "Labels_MEDIA_INFO_RESOLUTION",
                    language_index,
                ),
                _format_resolution(
                    metadata.video,
                ),
            ),
            _create_summary_card(
                get_localized_text(
                    "Labels_MEDIA_INFO_FRAME_RATE",
                    language_index,
                ),
                _format_frame_rate(
                    (metadata.video.frame_rate if metadata.video is not None else None),
                    language_index,
                ),
            ),
            _create_summary_card(
                get_localized_text(
                    "Labels_MEDIA_INFO_AUDIO_SUMMARY",
                    language_index,
                ),
                _format_audio_summary(
                    metadata.audio,
                    language_index,
                ),
            ),
        )
    )

    streams_template = get_localized_text(
        "Texts_MEDIA_STREAMS_SUMMARY",
        language_index,
    )
    streams_value = streams_template.format(
        video=metadata.video_stream_count,
        audio=metadata.audio_stream_count,
    )

    rows = "".join(
        (
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_FILENAME",
                    language_index,
                ),
                metadata.filename,
                code=True,
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_CONTAINER",
                    language_index,
                ),
                _format_container(
                    metadata.format_name,
                    metadata.format_long_name,
                ),
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_FILE_SIZE",
                    language_index,
                ),
                _format_file_size(
                    metadata.size_bytes,
                    language_index,
                ),
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_BIT_RATE",
                    language_index,
                ),
                _format_bit_rate(
                    metadata.bit_rate,
                    language_index,
                ),
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_STREAMS",
                    language_index,
                ),
                streams_value,
            ),
        )
    )

    return f"""
<section class="media-overview-content">
    {compatibility_state_marker}
    <div class="media-summary-grid">{summary_cards}</div>
    <div class="media-info-table">{rows}</div>
</section>
""".strip()
