"""
File: __init__.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Public media-information helpers for TACME.

License: MIT License
"""

from tacme.media.formatting import (
    create_audio_stream_html,
    create_media_compatibility_html,
    create_media_overview_html,
    create_media_unavailable_html,
    create_video_stream_html,
)
from tacme.media.metadata import (
    AudioStreamMetadata,
    MediaMetadata,
    MediaMetadataError,
    MediaProbeResult,
    VideoStreamMetadata,
    probe_media_file,
    read_media_metadata,
)
from tacme.media.validation import (
    MediaCompatibilityCheck,
    MediaCompatibilityResult,
    validate_media_for_tacme,
)

__all__ = [
    "AudioStreamMetadata",
    "MediaCompatibilityCheck",
    "MediaCompatibilityResult",
    "MediaMetadata",
    "MediaMetadataError",
    "MediaProbeResult",
    "VideoStreamMetadata",
    "create_audio_stream_html",
    "create_media_compatibility_html",
    "create_media_overview_html",
    "create_media_unavailable_html",
    "create_video_stream_html",
    "probe_media_file",
    "read_media_metadata",
    "validate_media_for_tacme",
]
