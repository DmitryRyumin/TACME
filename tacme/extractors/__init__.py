"""
Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Public feature extractor interfaces for the TACME inference pipeline.

License: MIT License
"""

from tacme.extractors.aggregation import (
    EXPECTED_AGGREGATED_LENGTH,
    FeatureAggregationError,
    mean_std_seq,
)
from tacme.extractors.audio import (
    DEFAULT_AUDIO_MODEL,
    EXPECTED_AUDIO_DIM,
    AudioExtractionError,
    EmoWav2Vec2Extractor,
)
from tacme.extractors.text import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_TEXT_MODEL,
    EXPECTED_TEXT_DIM,
    EmoRoBERTaExtractor,
    TextExtractionError,
)
from tacme.extractors.vision import (
    DEFAULT_VISION_MODEL,
    EXPECTED_VISION_DIM,
    CLIPSceneExtractor,
    VisionExtractionError,
)

__all__ = [
    "DEFAULT_AUDIO_MODEL",
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_TEXT_MODEL",
    "DEFAULT_VISION_MODEL",
    "EXPECTED_AGGREGATED_LENGTH",
    "EXPECTED_AUDIO_DIM",
    "EXPECTED_TEXT_DIM",
    "EXPECTED_VISION_DIM",
    "AudioExtractionError",
    "CLIPSceneExtractor",
    "EmoRoBERTaExtractor",
    "EmoWav2Vec2Extractor",
    "FeatureAggregationError",
    "TextExtractionError",
    "VisionExtractionError",
    "mean_std_seq",
]
