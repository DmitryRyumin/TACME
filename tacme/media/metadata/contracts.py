"""
File: contracts.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Typed metadata contracts and error classes for media inspection.

License: MIT License
"""

from dataclasses import dataclass


class MediaMetadataError(RuntimeError):
    """Raised when media metadata cannot be extracted."""


@dataclass(frozen=True, slots=True)
class VideoStreamMetadata:
    """Metadata for the primary video stream."""

    codec_name: str | None
    codec_long_name: str | None
    profile: str | None
    width: int | None
    height: int | None
    sample_aspect_ratio: str | None
    display_aspect_ratio: str | None
    frame_rate: float | None
    pixel_format: str | None
    bit_rate: float | None
    color_range: str | None
    color_space: str | None
    color_transfer: str | None
    color_primaries: str | None


@dataclass(frozen=True, slots=True)
class AudioStreamMetadata:
    """Metadata for the primary audio stream."""

    codec_name: str | None
    codec_long_name: str | None
    profile: str | None
    sample_rate: int | None
    num_channels: int | None
    channel_layout: str | None
    sample_format: str | None
    bit_rate: float | None


@dataclass(frozen=True, slots=True)
class MediaMetadata:
    """Container and primary-stream metadata for one media file."""

    filename: str
    size_bytes: int
    duration_seconds: float | None
    bit_rate: float | None
    format_name: str | None
    format_long_name: str | None
    video_stream_count: int
    audio_stream_count: int
    video: VideoStreamMetadata | None
    audio: AudioStreamMetadata | None


@dataclass(frozen=True, slots=True)
class MediaProbeResult:
    """Result of one ffprobe metadata extraction attempt."""

    metadata: MediaMetadata | None
    error: str | None = None

    @property
    def is_available(self) -> bool:
        """Return whether media metadata was extracted successfully."""

        return self.metadata is not None
