"""
File: __init__.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Backward-compatible public API for media metadata extraction.

License: MIT License
"""

from tacme.media.metadata.contracts import (
    AudioStreamMetadata as AudioStreamMetadata,
    MediaMetadata as MediaMetadata,
    MediaMetadataError as MediaMetadataError,
    MediaProbeResult as MediaProbeResult,
    VideoStreamMetadata as VideoStreamMetadata,
)
from tacme.media.metadata.probing import (
    FFPROBE_TIMEOUT_SECONDS as FFPROBE_TIMEOUT_SECONDS,
    probe_media_file as probe_media_file,
    read_media_metadata as read_media_metadata,
)

# Preserve the historical public class and function paths for introspection
# and compatibility with serialized instances.
MediaMetadataError.__module__ = __name__
VideoStreamMetadata.__module__ = __name__
AudioStreamMetadata.__module__ = __name__
MediaMetadata.__module__ = __name__
MediaProbeResult.__module__ = __name__
read_media_metadata.__module__ = __name__
probe_media_file.__module__ = __name__

__all__ = [
    "AudioStreamMetadata",
    "MediaMetadata",
    "MediaMetadataError",
    "MediaProbeResult",
    "VideoStreamMetadata",
    "probe_media_file",
    "read_media_metadata",
]
