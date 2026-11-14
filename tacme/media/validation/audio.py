"""
File: validation_audio.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: A4.2 audio-only stream validation and full decoding preflight.

License: MIT License
"""

from functools import lru_cache
from pathlib import Path
import subprocess
from typing import Literal

from tacme.media.metadata import probe_media_file
from tacme.media.validation.access import LOGGER, check_video_file_access
from tacme.media.validation.compatibility import classify_media_streams, validate_media_for_tacme
from tacme.media.validation.container import _is_truncated_mp4

type AudioFileStatus = Literal[
    "empty",
    "empty_file",
    "missing",
    "unreadable",
    "invalid_container",
    "truncated_file",
    "probe_unavailable",
    "decoder_unavailable",
    "missing_audio_stream",
    "not_audio_only",
    "invalid_audio_metadata",
    "undecodable_audio",
    "corrupted_audio",
    "available",
]


@lru_cache(maxsize=32)
def _inspect_audio_only_streams(
    file_path: str,
    size_bytes: int,
    modified_ns: int,
) -> AudioFileStatus:
    """Validate ffprobe metadata for an audio-only input, without decoding it."""

    probe = probe_media_file(file_path)
    if not probe.is_available:
        detail = probe.error or "unknown ffprobe error"
        if detail.startswith("Unable to execute ffprobe:"):
            LOGGER.warning("Unable to run the audio media inspector for %s", file_path)
            LOGGER.debug("ffprobe diagnostic: %s", detail)
            return "probe_unavailable"

        LOGGER.debug("Audio container rejected: %s; ffprobe: %s", file_path, detail)
        return "invalid_container"

    mode = classify_media_streams(probe)
    if mode in {"video_only", "no_streams"}:
        return "missing_audio_stream"
    if mode == "audiovisual":
        return "not_audio_only"
    if mode != "audio_only" or not validate_media_for_tacme(probe, allow_audio_only=True).is_ready:
        return "invalid_audio_metadata"

    return "available"


@lru_cache(maxsize=32)
def _check_first_decodable_audio(
    file_path: str,
    size_bytes: int,
    modified_ns: int,
) -> AudioFileStatus:
    """Require at least one decodable audio frame, without buffering the full waveform."""

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-xerror",
        "-err_detect",
        "explode",
        "-i",
        file_path,
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-frames:a",
        "1",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "s16le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=15.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        LOGGER.warning("First-frame audio inspection unavailable: %s", type(error).__name__)
        return "decoder_unavailable"

    if completed.returncode != 0 or len(completed.stdout) < 2 or len(completed.stdout) % 2 != 0:
        LOGGER.debug("Audio has no decodable samples: %s", file_path)
        LOGGER.debug("Audio decoder details: %s", completed.stderr.decode("utf-8", "replace")[:2000])
        return "undecodable_audio"

    return "available"


@lru_cache(maxsize=32)
def _check_full_audio_decode(
    file_path: str,
    size_bytes: int,
    modified_ns: int,
) -> AudioFileStatus:
    """Decode the entire first audio stream, rejecting errors beyond its first frame."""

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-xerror",
        "-err_detect",
        "explode",
        "-i",
        file_path,
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=60.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        LOGGER.warning("Full audio inspection unavailable: %s", type(error).__name__)
        return "decoder_unavailable"

    if completed.returncode != 0:
        LOGGER.debug("Audio contains damaged samples: %s", file_path)
        LOGGER.debug("Audio decoder details: %s", completed.stderr.decode("utf-8", "replace")[:2000])
        return "corrupted_audio"

    return "available"


def check_audio_only_container(media_path: str | None) -> AudioFileStatus:
    """A4.2: validate a physical audio-only stream, without changing video preflight.

    A missing video stream is valid. An existing but undecodable audio stream is
    not. Audio-only preflight does not infer speech presence or transcription quality.
    """

    access = check_video_file_access(media_path)
    if access == "empty":
        return "empty"
    if access == "empty_file":
        return "empty_file"
    if access == "missing":
        return "missing"
    if access != "available":
        return "unreadable"
    if not media_path:
        return "empty"

    path = Path(media_path)
    try:
        info = path.stat()
    except FileNotFoundError, NotADirectoryError:
        return "missing"
    except OSError, ValueError:
        return "unreadable"

    if _is_truncated_mp4(path):
        return "truncated_file"

    key = (str(path), info.st_size, info.st_mtime_ns)
    metadata_status = _inspect_audio_only_streams(*key)
    if metadata_status != "available":
        if metadata_status == "probe_unavailable":
            _inspect_audio_only_streams.cache_clear()
        return metadata_status

    first = _check_first_decodable_audio(*key)
    if first != "available":
        if first == "decoder_unavailable":
            _check_first_decodable_audio.cache_clear()
        return first

    complete = _check_full_audio_decode(*key)
    if complete == "decoder_unavailable":
        _check_full_audio_decode.cache_clear()
    return complete


__all__ = ["AudioFileStatus", "check_audio_only_container"]
