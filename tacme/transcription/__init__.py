"""
File: __init__.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Backward-compatible Whisper transcription public interfaces.

License: MIT License
"""

from tacme.transcription.audio import (
    _decode_media_waveform as _decode_media_waveform,
    _resolve_ffmpeg as _resolve_ffmpeg,
    _validate_media_path as _validate_media_path,
)
from tacme.transcription.contracts import (
    DEFAULT_SAMPLE_RATE as DEFAULT_SAMPLE_RATE,
    DEFAULT_WHISPER_MODEL as DEFAULT_WHISPER_MODEL,
    WHISPER_CHUNK_SECONDS as WHISPER_CHUNK_SECONDS,
    WhisperTranscription as WhisperTranscription,
    WhisperTranscriptionError as WhisperTranscriptionError,
)
from tacme.transcription.language import (
    _WHISPER_LANGUAGE_NAMES as _WHISPER_LANGUAGE_NAMES,
    _WhisperLanguageDetection as _WhisperLanguageDetection,
)
from tacme.transcription.transcriber import (
    WhisperTranscriber as WhisperTranscriber,
    _normalize_transcription as _normalize_transcription,
    _WhisperFeatureExtractorProtocol as _WhisperFeatureExtractorProtocol,
    _WhisperProcessorProtocol as _WhisperProcessorProtocol,
)

# Preserve original fully-qualified paths for existing pickle consumers.
for _legacy_class in (
    WhisperTranscriptionError,
    WhisperTranscription,
    _WhisperLanguageDetection,
    _WhisperFeatureExtractorProtocol,
    _WhisperProcessorProtocol,
    WhisperTranscriber,
):
    _legacy_class.__module__ = __name__
del _legacy_class


__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_WHISPER_MODEL",
    "WHISPER_CHUNK_SECONDS",
    "WhisperTranscriber",
    "WhisperTranscription",
    "WhisperTranscriptionError",
]
