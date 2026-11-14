"""
File: runtime.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Process-wide shared TACME inference engine and its lock.

License: MIT License
"""

from threading import RLock

from tacme.inference.engine import TacmeInferenceEngine

_INFERENCE_ENGINE_LOCK = RLock()

_INFERENCE_ENGINE: TacmeInferenceEngine | None = None


def get_tacme_inference_engine() -> TacmeInferenceEngine:
    """Return the shared process-wide TACME inference engine."""

    global _INFERENCE_ENGINE

    with _INFERENCE_ENGINE_LOCK:
        if _INFERENCE_ENGINE is None:
            _INFERENCE_ENGINE = TacmeInferenceEngine()

        return _INFERENCE_ENGINE


def preload_tacme_inference_engine() -> TacmeInferenceEngine:
    """Load and return the shared TACME inference engine."""

    return get_tacme_inference_engine()
