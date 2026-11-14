"""
File: validation.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Public validation API and ordered video preflight pipeline.

License: MIT License
"""

from pathlib import Path

from tacme.media.validation.access import (
    VideoFileStatus as VideoFileStatus,
    check_video_file_access as check_video_file_access,
)
from tacme.media.validation.audio import (
    AudioFileStatus as AudioFileStatus,
    check_audio_only_container as check_audio_only_container,
)
from tacme.media.validation.compatibility import (
    MediaCompatibilityCheck as MediaCompatibilityCheck,
    MediaCompatibilityResult as MediaCompatibilityResult,
    MediaStreamMode as MediaStreamMode,
    classify_media_streams as classify_media_streams,
    validate_media_for_tacme as validate_media_for_tacme,
)
from tacme.media.validation.container import _is_truncated_mp4, _probe_video_container
from tacme.media.validation.decode import (
    _check_first_decodable_frame,
    _check_full_video_decode,
)


def check_video_container(video_path: str | None) -> VideoFileStatus:
    """A2.2. Reject a container ffprobe cannot inspect before running models.

    Keep A1/A2.1 checks cheap and separate from this metadata-level check.
    A2.4 additionally checks that at least one video frame can be decoded.
    A2.5 additionally rejects corrupt frames anywhere in the stream.
    """

    access = check_video_file_access(video_path)
    if access != "available":
        return access
    if not video_path:
        return "empty"

    try:
        info = Path(video_path).stat()
    except FileNotFoundError, NotADirectoryError:
        return "missing"
    except OSError, ValueError:
        return "unreadable"

    # A2.3: ffprobe can accept a fast-start MP4 with a complete moov atom
    # even when its media-data (mdat) box is truncated. Verify box extents
    # before relying on metadata alone.
    if _is_truncated_mp4(Path(video_path)):
        return "truncated_file"

    result = _probe_video_container(
        str(Path(video_path)),
        info.st_size,
        info.st_mtime_ns,
    )
    if result != "available":
        if result == "probe_unavailable":
            # Allow transient failures to be retried without restarting TACME.
            _probe_video_container.cache_clear()
        return result

    decoded = _check_first_decodable_frame(
        str(Path(video_path)),
        info.st_size,
        info.st_mtime_ns,
    )
    if decoded != "available":
        if decoded == "decoder_unavailable":
            _check_first_decodable_frame.cache_clear()
        return decoded

    # A2.5: do not declare an upload ready after only its first good frame.
    complete = _check_full_video_decode(
        str(Path(video_path)),
        info.st_size,
        info.st_mtime_ns,
    )
    if complete == "decoder_unavailable":
        _check_full_video_decode.cache_clear()
    return complete


__all__ = [
    "AudioFileStatus",
    "MediaCompatibilityCheck",
    "MediaCompatibilityResult",
    "MediaStreamMode",
    "VideoFileStatus",
    "check_audio_only_container",
    "check_video_container",
    "check_video_file_access",
    "classify_media_streams",
    "validate_media_for_tacme",
]
