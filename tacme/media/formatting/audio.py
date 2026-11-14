"""
File: audio.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Audio-stream HTML for TACME media information.

License: MIT License
"""

from html import escape

from tacme.localization import get_localized_text
from tacme.media.formatting.components import _create_info_row, create_media_unavailable_html
from tacme.media.formatting.values import (
    _format_bit_rate,
    _format_codec,
    _format_optional_value,
    _format_sample_rate,
)
from tacme.media.metadata import MediaProbeResult


def create_audio_stream_html(
    probe_result: MediaProbeResult | None,
    language_index: int,
) -> str:
    """Create the Audio tab content."""

    if probe_result is None or probe_result.metadata is None:
        return create_media_unavailable_html(
            language_index,
        )

    audio = probe_result.metadata.audio

    if audio is None:
        message = escape(
            get_localized_text(
                "Texts_MEDIA_NO_AUDIO_STREAM",
                language_index,
            )
        )

        return f'<section class="media-info-empty-state"><p>{message}</p></section>'

    rows = "".join(
        (
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_CODEC",
                    language_index,
                ),
                _format_codec(
                    audio.codec_name,
                    audio.codec_long_name,
                ),
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_PROFILE",
                    language_index,
                ),
                _format_optional_value(
                    audio.profile,
                ),
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_SAMPLE_RATE",
                    language_index,
                ),
                _format_sample_rate(
                    audio.sample_rate,
                    language_index,
                ),
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_CHANNELS",
                    language_index,
                ),
                _format_optional_value(
                    audio.num_channels,
                ),
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_CHANNEL_LAYOUT",
                    language_index,
                ),
                _format_optional_value(
                    audio.channel_layout,
                ),
                code=bool(
                    audio.channel_layout,
                ),
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_SAMPLE_FORMAT",
                    language_index,
                ),
                _format_optional_value(
                    audio.sample_format,
                ),
                code=bool(
                    audio.sample_format,
                ),
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_BIT_RATE",
                    language_index,
                ),
                _format_bit_rate(
                    audio.bit_rate,
                    language_index,
                ),
            ),
        )
    )

    return f'<section class="media-info-table">{rows}</section>'
