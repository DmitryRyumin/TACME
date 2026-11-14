"""
File: media.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Path validation, media stream selection, and FFmpeg/ffprobe lookup.

License: MIT License
"""

from pathlib import Path
import shutil
from typing import Literal

from tacme.media.metadata import probe_media_file
from tacme.media.validation import check_audio_only_container, classify_media_streams
from tacme.pipeline.contracts import PipelineError


def _validate_media_path(
    media_path: str | Path,
) -> Path:
    """Validate one input media path."""

    path = Path(
        media_path,
    ).expanduser()

    if not path.exists():
        msg = f"Input media file does not exist: {path}"
        raise PipelineError(msg)

    if not path.is_file():
        msg = f"Input media path is not a file: {path}"
        raise PipelineError(msg)

    return path.resolve()


def _find_executable(
    name: str,
) -> str:
    """Return one required executable from PATH."""

    executable = shutil.which(
        name,
    )

    if executable is None:
        msg = f"Required executable {name!r} was not found in PATH."
        raise PipelineError(msg)

    return executable


def _resolve_pipeline_stream_mode(
    path: Path,
    *,
    allow_audio_only: bool,
) -> Literal["audiovisual", "audio_only"]:
    """Select an input mode without changing the default audiovisual pipeline.

    Audio-only is opt-in and requires the complete A4.2 decoding preflight.
    An existing but invalid audio stream must never be treated as missing.
    """

    if not allow_audio_only:
        return "audiovisual"

    probe = probe_media_file(path)
    mode = classify_media_streams(probe)

    if mode == "audiovisual":
        return "audiovisual"

    if mode == "audio_only":
        status = check_audio_only_container(str(path))
        if status != "available":
            msg = f"Audio-only media validation failed: {status}."
            raise PipelineError(msg)
        return "audio_only"

    if mode == "unavailable":
        msg = f"Unable to inspect input media streams: {probe.error or 'unknown ffprobe error'}"
        raise PipelineError(msg)

    msg = f"TACME requires an audio stream; detected media mode: {mode}."
    raise PipelineError(msg)
