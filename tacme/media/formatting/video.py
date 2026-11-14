"""
File: video.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Video-stream HTML for TACME media information.

License: MIT License
"""

from html import escape

from tacme.localization import get_localized_text
from tacme.media.formatting.components import _create_info_row, create_media_unavailable_html
from tacme.media.formatting.values import (
    _format_bit_rate,
    _format_codec,
    _format_frame_rate,
    _format_optional_value,
    _format_resolution,
)
from tacme.media.metadata import MediaProbeResult


def create_video_stream_html(
    probe_result: MediaProbeResult | None,
    language_index: int,
) -> str:
    """Create the Video tab content."""

    if probe_result is None or probe_result.metadata is None:
        return create_media_unavailable_html(
            language_index,
        )

    video = probe_result.metadata.video

    if video is None:
        message = escape(
            get_localized_text(
                "Texts_MEDIA_NO_VIDEO_STREAM",
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
                    video.codec_name,
                    video.codec_long_name,
                ),
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_PROFILE",
                    language_index,
                ),
                _format_optional_value(
                    video.profile,
                ),
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_RESOLUTION",
                    language_index,
                ),
                _format_resolution(
                    video,
                ),
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_ASPECT_RATIO",
                    language_index,
                ),
                _format_optional_value(
                    video.display_aspect_ratio,
                ),
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_FRAME_RATE",
                    language_index,
                ),
                _format_frame_rate(
                    video.frame_rate,
                    language_index,
                ),
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_PIXEL_FORMAT",
                    language_index,
                ),
                _format_optional_value(
                    video.pixel_format,
                ),
                code=bool(
                    video.pixel_format,
                ),
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_BIT_RATE",
                    language_index,
                ),
                _format_bit_rate(
                    video.bit_rate,
                    language_index,
                ),
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_COLOR_SPACE",
                    language_index,
                ),
                _format_optional_value(
                    video.color_space,
                ),
                code=bool(
                    video.color_space,
                ),
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_COLOR_RANGE",
                    language_index,
                ),
                _format_optional_value(
                    video.color_range,
                ),
                code=bool(
                    video.color_range,
                ),
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_COLOR_TRANSFER",
                    language_index,
                ),
                _format_optional_value(
                    video.color_transfer,
                ),
                code=bool(
                    video.color_transfer,
                ),
            ),
            _create_info_row(
                get_localized_text(
                    "Labels_MEDIA_INFO_COLOR_PRIMARIES",
                    language_index,
                ),
                _format_optional_value(
                    video.color_primaries,
                ),
                code=bool(
                    video.color_primaries,
                ),
            ),
        )
    )

    return f'<section class="media-info-table">{rows}</section>'
