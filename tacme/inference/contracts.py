"""
File: contracts.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Public prediction contracts, labels, and inference exceptions.

License: MIT License
"""

from dataclasses import dataclass
from typing import Final

from tacme.behavior import BehaviorDescription
from tacme.explainability.attribution import FeatureAttributionBundle
from tacme.pipeline import SceneFrameArtifacts
from tacme.transcription import WhisperTranscription

EMOTION_LABELS: Final[tuple[str, ...]] = (
    "Neutral",
    "Anger",
    "Disgust",
    "Fear",
    "Happiness",
    "Sadness",
    "Surprise",
)
PERSONALITY_LABELS: Final[tuple[str, ...]] = (
    "OPE",
    "CON",
    "EXT",
    "AGR",
    "NNEU",
)
AMBIVALENCE_LABELS: Final[tuple[str, ...]] = (
    "Absence",
    "Presence",
)
EXPECTED_EMOTION_DIM: Final = len(
    EMOTION_LABELS,
)
EXPECTED_PERSONALITY_DIM: Final = len(
    PERSONALITY_LABELS,
)
EXPECTED_AMBIVALENCE_DIM: Final = len(
    AMBIVALENCE_LABELS,
)


class TacmeInferenceError(RuntimeError):
    """Raised when TACME end-to-end inference fails."""


@dataclass(
    frozen=True,
    slots=True,
)
class ClassificationPrediction:
    """One categorical TACME prediction."""

    label: str
    class_index: int
    confidence: float
    probabilities: tuple[float, ...]


@dataclass(
    frozen=True,
    slots=True,
)
class PersonalityPrediction:
    """Five apparent personality trait scores."""

    labels: tuple[str, ...]
    scores: tuple[float, ...]


@dataclass(
    frozen=True,
    slots=True,
)
class AmbivalencePrediction:
    """Binary ambivalence prediction."""

    label: str
    class_index: int
    confidence: float
    absence_probability: float
    presence_probability: float


@dataclass(
    frozen=True,
    slots=True,
)
class TacmeInferenceResult:
    """Human-consumable TACME predictions for one video."""

    emotion: ClassificationPrediction
    personality: PersonalityPrediction
    ambivalence: AmbivalencePrediction


@dataclass(
    frozen=True,
    slots=True,
)
class TacmeInferenceRun:
    """One TACME prediction together with retained multimodal inference artifacts."""

    result: TacmeInferenceResult
    scene_artifacts: SceneFrameArtifacts
    transcription: WhisperTranscription
    behavior_description: BehaviorDescription
    feature_attribution: FeatureAttributionBundle
