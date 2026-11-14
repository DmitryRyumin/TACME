"""
File: contracts.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Speech transcription defaults and stable result contracts.

License: MIT License
"""

from dataclasses import dataclass
from typing import Final

DEFAULT_WHISPER_MODEL: Final = "openai/whisper-small"


DEFAULT_SAMPLE_RATE: Final = 16_000


WHISPER_CHUNK_SECONDS: Final = 30


class WhisperTranscriptionError(RuntimeError):
    """Raised when TACME speech transcription fails."""


@dataclass(
    frozen=True,
    slots=True,
)
class WhisperTranscription:
    """Speech transcription together with Whisper language metadata."""

    text: str
    language_code: str | None
    language_name: str | None
    language_confidence: float | None
