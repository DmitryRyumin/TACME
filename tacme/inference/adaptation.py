"""
File: adaptation.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Validation and adaptation of TACME prediction tensors.

License: MIT License
"""

from collections.abc import Mapping

import torch
from torch import Tensor

from tacme.inference.contracts import (
    AMBIVALENCE_LABELS,
    EMOTION_LABELS,
    EXPECTED_AMBIVALENCE_DIM,
    EXPECTED_EMOTION_DIM,
    EXPECTED_PERSONALITY_DIM,
    PERSONALITY_LABELS,
    AmbivalencePrediction,
    ClassificationPrediction,
    PersonalityPrediction,
    TacmeInferenceError,
    TacmeInferenceResult,
)


def _validate_output_tensor(
    name: str,
    tensor: Tensor,
    *,
    expected_dim: int,
) -> Tensor:
    """Validate one single-sample TACME output tensor."""

    expected_shape = (
        1,
        expected_dim,
    )

    if (
        tuple(
            tensor.shape,
        )
        != expected_shape
    ):
        msg = f"Unexpected TACME output shape for {name!r}: {tuple(tensor.shape)}; expected {expected_shape}."
        raise TacmeInferenceError(msg)

    if tensor.dtype != torch.float32:
        msg = f"Unexpected TACME output dtype for {name!r}: {tensor.dtype}; expected torch.float32."
        raise TacmeInferenceError(msg)

    if not bool(
        torch.isfinite(
            tensor,
        )
        .all()
        .item()
    ):
        msg = f"TACME output {name!r} contains NaN or infinite values."
        raise TacmeInferenceError(msg)

    return tensor


def _require_output_tensor(
    outputs: Mapping[str, Tensor | None],
    name: str,
    *,
    expected_dim: int,
) -> Tensor:
    """Return and validate one required TACME model output."""

    tensor = outputs.get(
        name,
    )

    if tensor is None:
        msg = f"TACME returned no required output {name!r}."
        raise TacmeInferenceError(msg)

    return _validate_output_tensor(
        name,
        tensor,
        expected_dim=expected_dim,
    )


def _tensor_to_float_tuple(
    tensor: Tensor,
) -> tuple[float, ...]:
    """Convert one one-dimensional tensor into immutable Python floats."""

    if tensor.ndim != 1:
        msg = f"Expected a one-dimensional tensor, got shape {tuple(tensor.shape)}."
        raise TacmeInferenceError(msg)

    return tuple(
        float(
            value,
        )
        for value in (tensor.detach().cpu().tolist())
    )


def _adapt_emotion_prediction(
    emotion_logits: Tensor,
) -> ClassificationPrediction:
    """Convert TACME emotion logits into a categorical prediction."""

    probabilities_tensor = torch.softmax(
        emotion_logits[0],
        dim=0,
    )

    class_index = int(
        torch.argmax(
            probabilities_tensor,
        )
        .detach()
        .cpu()
        .item()
    )

    probabilities = _tensor_to_float_tuple(
        probabilities_tensor,
    )

    return ClassificationPrediction(
        label=EMOTION_LABELS[class_index],
        class_index=class_index,
        confidence=probabilities[class_index],
        probabilities=probabilities,
    )


def _adapt_personality_prediction(
    personality_scores: Tensor,
) -> PersonalityPrediction:
    """Convert TACME personality output into five trait scores."""

    scores = _tensor_to_float_tuple(
        personality_scores[0],
    )

    return PersonalityPrediction(
        labels=PERSONALITY_LABELS,
        scores=scores,
    )


def _adapt_ambivalence_prediction(
    ah_logits: Tensor,
) -> AmbivalencePrediction:
    """Convert TACME ambivalence logits into a binary prediction."""

    probabilities_tensor = torch.softmax(
        ah_logits[0],
        dim=0,
    )

    class_index = int(
        torch.argmax(
            probabilities_tensor,
        )
        .detach()
        .cpu()
        .item()
    )

    probabilities = _tensor_to_float_tuple(
        probabilities_tensor,
    )

    return AmbivalencePrediction(
        label=AMBIVALENCE_LABELS[class_index],
        class_index=class_index,
        confidence=probabilities[class_index],
        absence_probability=probabilities[0],
        presence_probability=probabilities[1],
    )


def adapt_tacme_outputs(
    outputs: Mapping[str, Tensor | None],
) -> TacmeInferenceResult:
    """Convert raw TACME model outputs into the public inference result."""

    emotion_logits = _require_output_tensor(
        outputs,
        "emotion_logits",
        expected_dim=EXPECTED_EMOTION_DIM,
    )

    personality_scores = _require_output_tensor(
        outputs,
        "personality_scores",
        expected_dim=EXPECTED_PERSONALITY_DIM,
    )

    ah_logits = _require_output_tensor(
        outputs,
        "ah_logits",
        expected_dim=EXPECTED_AMBIVALENCE_DIM,
    )

    emotion = _adapt_emotion_prediction(
        emotion_logits,
    )

    personality = _adapt_personality_prediction(
        personality_scores,
    )

    ambivalence = _adapt_ambivalence_prediction(
        ah_logits,
    )

    return TacmeInferenceResult(
        emotion=emotion,
        personality=personality,
        ambivalence=ambivalence,
    )
