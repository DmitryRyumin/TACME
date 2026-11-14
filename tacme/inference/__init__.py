"""
File: __init__.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Backward-compatible public API for TACME inference.

License: MIT License
"""

from tacme.explainability.attribution import FeatureAttributionBundle as FeatureAttributionBundle
from tacme.inference.adaptation import (
    _adapt_ambivalence_prediction as _adapt_ambivalence_prediction,
    _adapt_emotion_prediction as _adapt_emotion_prediction,
    _adapt_personality_prediction as _adapt_personality_prediction,
    _require_output_tensor as _require_output_tensor,
    _tensor_to_float_tuple as _tensor_to_float_tuple,
    _validate_output_tensor as _validate_output_tensor,
    adapt_tacme_outputs as adapt_tacme_outputs,
)
from tacme.inference.contracts import (
    AMBIVALENCE_LABELS as AMBIVALENCE_LABELS,
    EMOTION_LABELS as EMOTION_LABELS,
    EXPECTED_AMBIVALENCE_DIM as EXPECTED_AMBIVALENCE_DIM,
    EXPECTED_EMOTION_DIM as EXPECTED_EMOTION_DIM,
    EXPECTED_PERSONALITY_DIM as EXPECTED_PERSONALITY_DIM,
    PERSONALITY_LABELS as PERSONALITY_LABELS,
    AmbivalencePrediction as AmbivalencePrediction,
    ClassificationPrediction as ClassificationPrediction,
    PersonalityPrediction as PersonalityPrediction,
    TacmeInferenceError as TacmeInferenceError,
    TacmeInferenceResult as TacmeInferenceResult,
    TacmeInferenceRun as TacmeInferenceRun,
)
from tacme.inference.engine import TacmeInferenceEngine as TacmeInferenceEngine
from tacme.inference.runtime import (
    _INFERENCE_ENGINE_LOCK as _INFERENCE_ENGINE_LOCK,
    get_tacme_inference_engine as get_tacme_inference_engine,
    preload_tacme_inference_engine as preload_tacme_inference_engine,
)

# Preserve historical names used for serialization and introspection.
TacmeInferenceError.__module__ = __name__
ClassificationPrediction.__module__ = __name__
PersonalityPrediction.__module__ = __name__
AmbivalencePrediction.__module__ = __name__
TacmeInferenceResult.__module__ = __name__
TacmeInferenceRun.__module__ = __name__
_validate_output_tensor.__module__ = __name__
_require_output_tensor.__module__ = __name__
_tensor_to_float_tuple.__module__ = __name__
_adapt_emotion_prediction.__module__ = __name__
_adapt_personality_prediction.__module__ = __name__
_adapt_ambivalence_prediction.__module__ = __name__
adapt_tacme_outputs.__module__ = __name__
TacmeInferenceEngine.__module__ = __name__
get_tacme_inference_engine.__module__ = __name__
preload_tacme_inference_engine.__module__ = __name__

__all__ = [
    "AMBIVALENCE_LABELS",
    "EMOTION_LABELS",
    "PERSONALITY_LABELS",
    "AmbivalencePrediction",
    "ClassificationPrediction",
    "FeatureAttributionBundle",
    "PersonalityPrediction",
    "TacmeInferenceEngine",
    "TacmeInferenceError",
    "TacmeInferenceResult",
    "TacmeInferenceRun",
    "adapt_tacme_outputs",
    "get_tacme_inference_engine",
    "preload_tacme_inference_engine",
]
