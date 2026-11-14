"""
File: probing.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Use ffprobe to inspect media and produce metadata results.

License: MIT License
"""

import json
from pathlib import Path
import subprocess
from typing import Final

from tacme.media.metadata.contracts import (
    MediaMetadata,
    MediaMetadataError,
    MediaProbeResult,
)
from tacme.media.metadata.parsing import (
    _as_mapping,
    _as_streams,
    _parse_audio_stream,
    _parse_optional_float,
    _parse_optional_str,
    _parse_video_stream,
    _select_duration,
)

FFPROBE_TIMEOUT_SECONDS: Final = 15.0


def read_media_metadata(
    media_path: str | Path,
) -> MediaMetadata:
    """Read container, video, and audio metadata using ffprobe."""

    path = Path(
        media_path,
    )

    if not path.is_file():
        msg = f"Media file does not exist: {path}"
        raise MediaMetadataError(msg)

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=FFPROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        msg = f"Unable to execute ffprobe: {error}"
        raise MediaMetadataError(msg) from error

    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown ffprobe error"
        msg = f"ffprobe failed: {detail}"
        raise MediaMetadataError(msg)

    try:
        payload = json.loads(
            completed.stdout,
        )
    except json.JSONDecodeError as error:
        msg = "ffprobe returned invalid JSON metadata."
        raise MediaMetadataError(msg) from error

    root = _as_mapping(payload)

    if root is None:
        msg = "ffprobe returned an unexpected metadata structure."
        raise MediaMetadataError(msg)

    format_data = (
        _as_mapping(
            root.get("format"),
        )
        or {}
    )

    streams = _as_streams(
        root.get("streams"),
    )

    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]

    primary_video = video_streams[0] if video_streams else None
    primary_audio = audio_streams[0] if audio_streams else None

    try:
        size_bytes = path.stat().st_size
    except OSError as error:
        msg = f"Unable to read media file information: {error}"
        raise MediaMetadataError(msg) from error

    return MediaMetadata(
        filename=path.name,
        size_bytes=size_bytes,
        duration_seconds=_select_duration(
            format_data,
            primary_video,
            primary_audio,
        ),
        bit_rate=_parse_optional_float(
            format_data.get("bit_rate"),
        ),
        format_name=_parse_optional_str(
            format_data.get("format_name"),
        ),
        format_long_name=_parse_optional_str(
            format_data.get("format_long_name"),
        ),
        video_stream_count=len(
            video_streams,
        ),
        audio_stream_count=len(
            audio_streams,
        ),
        video=(
            _parse_video_stream(
                primary_video,
            )
            if primary_video is not None
            else None
        ),
        audio=(
            _parse_audio_stream(
                primary_audio,
            )
            if primary_audio is not None
            else None
        ),
    )


def probe_media_file(
    media_path: str | Path,
) -> MediaProbeResult:
    """Probe one media file without propagating metadata errors."""

    try:
        metadata = read_media_metadata(
            media_path,
        )
    except MediaMetadataError as error:
        return MediaProbeResult(
            metadata=None,
            error=str(
                error,
            ),
        )

    return MediaProbeResult(
        metadata=metadata,
    )
