"""
File: parsing.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Parse values and stream information returned by ffprobe.

License: MIT License
"""

from collections.abc import Mapping
from math import gcd
from typing import Any

from tacme.media.metadata.contracts import AudioStreamMetadata, VideoStreamMetadata


def _parse_optional_float(
    value: object,
) -> float | None:
    """Parse one optional ffprobe numeric value as float."""

    if isinstance(
        value,
        bool,
    ):
        return None

    if not isinstance(
        value,
        (
            str,
            int,
            float,
        ),
    ):
        return None

    try:
        return float(
            value,
        )
    except ValueError:
        return None


def _parse_optional_int(
    value: object,
) -> int | None:
    """Parse one optional ffprobe numeric value as int."""

    if isinstance(
        value,
        bool,
    ):
        return None

    if isinstance(
        value,
        int,
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        if not value.is_integer():
            return None

        return int(
            value,
        )

    if isinstance(
        value,
        str,
    ):
        try:
            return int(
                value,
            )
        except ValueError:
            return None

    return None


def _parse_optional_str(value: object) -> str | None:
    """Parse one optional ffprobe value as a non-empty string."""

    if value is None:
        return None

    text = str(value).strip()

    return text or None


def _parse_frame_rate(value: object) -> float | None:
    """Parse an ffprobe rational frame-rate value."""

    text = _parse_optional_str(value)

    if not text:
        return None

    if "/" not in text:
        return _parse_optional_float(text)

    numerator_text, denominator_text = text.split(
        "/",
        maxsplit=1,
    )

    numerator = _parse_optional_float(numerator_text)
    denominator = _parse_optional_float(denominator_text)

    if numerator is None or denominator is None or denominator == 0.0:
        return None

    return numerator / denominator


def _as_mapping(value: object) -> Mapping[str, Any] | None:
    """Return a JSON object as a mapping when possible."""

    if isinstance(value, dict):
        return value

    return None


def _as_streams(value: object) -> list[Mapping[str, Any]]:
    """Return valid ffprobe stream objects."""

    if not isinstance(value, list):
        return []

    streams: list[Mapping[str, Any]] = []

    for item in value:
        stream = _as_mapping(item)

        if stream is not None:
            streams.append(stream)

    return streams


def _create_display_aspect_ratio(
    width: int | None,
    height: int | None,
) -> str | None:
    """Create a reduced display-aspect ratio from dimensions."""

    if not width or not height or width <= 0 or height <= 0:
        return None

    divisor = gcd(
        width,
        height,
    )

    return f"{width // divisor}:{height // divisor}"


def _parse_video_stream(
    stream: Mapping[str, Any],
) -> VideoStreamMetadata:
    """Convert one ffprobe video stream into typed metadata."""

    width = _parse_optional_int(
        stream.get("width"),
    )
    height = _parse_optional_int(
        stream.get("height"),
    )

    frame_rate = _parse_frame_rate(
        stream.get("avg_frame_rate"),
    )

    if frame_rate in (None, 0.0):
        frame_rate = _parse_frame_rate(
            stream.get("r_frame_rate"),
        )

    display_aspect_ratio = _parse_optional_str(
        stream.get("display_aspect_ratio"),
    )

    if not display_aspect_ratio or display_aspect_ratio == "0:1":
        display_aspect_ratio = _create_display_aspect_ratio(
            width,
            height,
        )

    return VideoStreamMetadata(
        codec_name=_parse_optional_str(
            stream.get("codec_name"),
        ),
        codec_long_name=_parse_optional_str(
            stream.get("codec_long_name"),
        ),
        profile=_parse_optional_str(
            stream.get("profile"),
        ),
        width=width,
        height=height,
        sample_aspect_ratio=_parse_optional_str(
            stream.get("sample_aspect_ratio"),
        ),
        display_aspect_ratio=display_aspect_ratio,
        frame_rate=frame_rate,
        pixel_format=_parse_optional_str(
            stream.get("pix_fmt"),
        ),
        bit_rate=_parse_optional_float(
            stream.get("bit_rate"),
        ),
        color_range=_parse_optional_str(
            stream.get("color_range"),
        ),
        color_space=_parse_optional_str(
            stream.get("color_space"),
        ),
        color_transfer=_parse_optional_str(
            stream.get("color_transfer"),
        ),
        color_primaries=_parse_optional_str(
            stream.get("color_primaries"),
        ),
    )


def _parse_audio_stream(
    stream: Mapping[str, Any],
) -> AudioStreamMetadata:
    """Convert one ffprobe audio stream into typed metadata."""

    return AudioStreamMetadata(
        codec_name=_parse_optional_str(
            stream.get("codec_name"),
        ),
        codec_long_name=_parse_optional_str(
            stream.get("codec_long_name"),
        ),
        profile=_parse_optional_str(
            stream.get("profile"),
        ),
        sample_rate=_parse_optional_int(
            stream.get("sample_rate"),
        ),
        num_channels=_parse_optional_int(
            stream.get("channels"),
        ),
        channel_layout=_parse_optional_str(
            stream.get("channel_layout"),
        ),
        sample_format=_parse_optional_str(
            stream.get("sample_fmt"),
        ),
        bit_rate=_parse_optional_float(
            stream.get("bit_rate"),
        ),
    )


def _select_duration(
    format_data: Mapping[str, Any],
    video_stream: Mapping[str, Any] | None,
    audio_stream: Mapping[str, Any] | None,
) -> float | None:
    """Return the best available media duration."""

    duration = _parse_optional_float(
        format_data.get("duration"),
    )

    if duration is not None and duration > 0:
        return duration

    for stream in (
        video_stream,
        audio_stream,
    ):
        if stream is None:
            continue

        duration = _parse_optional_float(
            stream.get("duration"),
        )

        if duration is not None and duration > 0:
            return duration

    return None
