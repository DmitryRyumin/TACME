"""
File: values.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Formatting values for TACME media-information HTML.

License: MIT License
"""

from tacme.localization import get_localized_text
from tacme.media.metadata import AudioStreamMetadata, VideoStreamMetadata


def _format_optional_value(
    value: object | None,
) -> str:
    """Format an optional metadata value."""

    return (
        "—"
        if value is None
        else str(
            value,
        )
    )


def _format_duration(
    seconds: float | None,
    language_index: int,
) -> str:
    """Format duration in seconds."""

    if seconds is None:
        return "—"

    unit = get_localized_text(
        "Units_SECONDS",
        language_index,
    )

    return f"{seconds:.2f} {unit}"


def _format_file_size(
    size_bytes: int,
    language_index: int,
) -> str:
    """Format a byte count using binary units."""

    if size_bytes < 1024:
        return f"{size_bytes} B"

    if size_bytes < 1024**2:
        unit = get_localized_text(
            "Units_KILOBYTES",
            language_index,
        )

        return f"{size_bytes / 1024:.1f} {unit}"

    if size_bytes < 1024**3:
        return f"{size_bytes / 1024**2:.2f} MB"

    return f"{size_bytes / 1024**3:.2f} GB"


def _format_bit_rate(
    bit_rate: float | None,
    language_index: int,
) -> str:
    """Format a bit rate using kbps or Mbps."""

    if bit_rate is None:
        return "—"

    if bit_rate >= 1_000_000:
        return f"{bit_rate / 1_000_000:.2f} Mbps"

    unit = get_localized_text(
        "Units_KILOBITS_PER_SECOND",
        language_index,
    )

    return f"{bit_rate / 1000:.0f} {unit}"


def _format_sample_rate(
    sample_rate: int | None,
    language_index: int,
) -> str:
    """Format an audio sample rate."""

    if sample_rate is None:
        return "—"

    unit = get_localized_text(
        "Units_HERTZ",
        language_index,
    )

    formatted_value = f"{sample_rate:,}"

    if language_index == 1:
        formatted_value = formatted_value.replace(
            ",",
            " ",
        )

    return f"{formatted_value} {unit}"


def _format_frame_rate(
    frame_rate: float | None,
    language_index: int,
) -> str:
    """Format a video frame rate."""

    if frame_rate is None:
        return "—"

    unit = get_localized_text(
        "Units_FRAMES_PER_SECOND",
        language_index,
    )

    return f"{frame_rate:.2f} {unit}"


def _format_resolution(
    video: VideoStreamMetadata | None,
) -> str:
    """Format video dimensions."""

    if video is None or video.width is None or video.height is None:
        return "—"

    return f"{video.width} \u00d7 {video.height}"


def _format_codec(
    codec_name: str | None,
    codec_long_name: str | None,
) -> str:
    """Format a codec name and its human-readable description."""

    if codec_name and codec_long_name:
        if codec_name.casefold() in codec_long_name.casefold():
            return codec_long_name

        return f"{codec_name} · {codec_long_name}"

    return _format_optional_value(
        codec_long_name or codec_name,
    )


def _format_container(
    format_name: str | None,
    format_long_name: str | None,
) -> str:
    """Format media container information."""

    if format_long_name:
        return format_long_name

    return _format_optional_value(
        format_name,
    )


def _format_audio_summary(
    audio: AudioStreamMetadata | None,
    language_index: int,
) -> str:
    """Format compact audio information for the overview cards."""

    if audio is None:
        return get_localized_text(
            "Texts_MEDIA_NO_AUDIO_SHORT",
            language_index,
        )

    if audio.sample_rate is None and audio.num_channels is None:
        return _format_codec(
            audio.codec_name,
            audio.codec_long_name,
        )

    parts: list[str] = []

    if audio.sample_rate is not None:
        unit = get_localized_text(
            "Units_KILOHERTZ",
            language_index,
        )

        parts.append(
            f"{audio.sample_rate / 1000:.1f} {unit}",
        )

    if audio.num_channels is not None:
        unit = get_localized_text(
            "Units_CHANNELS_SHORT",
            language_index,
        )

        parts.append(
            f"{audio.num_channels} {unit}",
        )

    return " · ".join(
        parts,
    )
