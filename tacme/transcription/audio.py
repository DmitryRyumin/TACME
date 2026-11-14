"""
File: audio.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: FFmpeg audio extraction and media path validation for Whisper.

License: MIT License
"""

from pathlib import Path
import shutil
import subprocess

import torch
from torch import Tensor

from tacme.transcription.contracts import WhisperTranscriptionError


def _validate_media_path(
    media_path: Path,
) -> Path:
    """Resolve and validate an input audio or video path."""

    path = media_path.expanduser().resolve()

    if not path.is_file():
        msg = f"Media file does not exist: {path}"
        raise WhisperTranscriptionError(msg)

    return path


def _resolve_ffmpeg() -> str:
    """Resolve the FFmpeg executable used for Whisper audio decoding."""

    executable = shutil.which(
        "ffmpeg",
    )

    if executable is None:
        raise WhisperTranscriptionError("FFmpeg is required for Whisper audio decoding but was not found in PATH.")

    return executable


def _decode_media_waveform(
    media_path: Path,
    *,
    target_sample_rate: int,
) -> Tensor:
    """Decode input media into a mono float32 waveform for Whisper."""

    path = _validate_media_path(
        media_path,
    )

    if target_sample_rate <= 0:
        msg = f"Invalid Whisper target sample rate: {target_sample_rate}."
        raise WhisperTranscriptionError(msg)

    ffmpeg = _resolve_ffmpeg()

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(target_sample_rate),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
        )

    except OSError as error:
        msg = f"Unable to execute FFmpeg for Whisper audio decoding from {path}: {error}"
        raise WhisperTranscriptionError(msg) from error

    if result.returncode != 0:
        error_message = result.stderr.decode(
            "utf-8",
            errors="replace",
        ).strip()

        if not error_message:
            error_message = f"FFmpeg exited with code {result.returncode}."

        msg = f"Unable to decode Whisper audio from {path}: {error_message}"
        raise WhisperTranscriptionError(msg)

    if not result.stdout:
        msg = f"Media file contains no decodable audio samples: {path}"
        raise WhisperTranscriptionError(msg)

    if len(result.stdout) % torch.float32.itemsize != 0:
        msg = f"FFmpeg returned an invalid float32 Whisper audio buffer with {len(result.stdout)} bytes."
        raise WhisperTranscriptionError(msg)

    buffer = bytearray(
        result.stdout,
    )

    waveform = torch.frombuffer(
        buffer,
        dtype=torch.float32,
    ).clone()

    if waveform.numel() == 0:
        msg = f"Decoded Whisper waveform is empty: {path}"
        raise WhisperTranscriptionError(msg)

    if not bool(
        torch.isfinite(
            waveform,
        )
        .all()
        .item()
    ):
        raise WhisperTranscriptionError("Decoded Whisper waveform contains NaN or infinite values.")

    return waveform.contiguous()
