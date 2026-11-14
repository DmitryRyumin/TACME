"""
File: check_inference.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Smoke test for TACME prediction adaptation and public inference results.

License: MIT License
"""

import math
import sys
from typing import Final

import torch
from torch import Tensor

from tacme.inference import (
    AMBIVALENCE_LABELS,
    EMOTION_LABELS,
    PERSONALITY_LABELS,
    TacmeInferenceError,
    TacmeInferenceResult,
    adapt_tacme_outputs,
)

PROBABILITY_TOLERANCE: Final = 1e-6

EXPECTED_EMOTION_CLASS: Final = 4
EXPECTED_EMOTION_LABEL: Final = "Happiness"

EXPECTED_AMBIVALENCE_CLASS: Final = 1
EXPECTED_AMBIVALENCE_LABEL: Final = "Presence"

EXPECTED_PERSONALITY_SCORES: Final = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
)


def _create_test_outputs() -> dict[str, Tensor | None]:
    """Create deterministic TACME task outputs for adaptation testing."""

    return {
        "emotion_logits": torch.tensor(
            [
                [
                    0.0,
                    1.0,
                    -1.0,
                    0.5,
                    3.0,
                    -2.0,
                    0.25,
                ],
            ],
            dtype=torch.float32,
        ),
        "personality_scores": torch.tensor(
            [
                EXPECTED_PERSONALITY_SCORES,
            ],
            dtype=torch.float32,
        ),
        "ah_logits": torch.tensor(
            [
                [
                    -1.0,
                    2.0,
                ],
            ],
            dtype=torch.float32,
        ),
    }


def _validate_probability_distribution(
    probabilities: tuple[float, ...],
    *,
    expected_length: int,
    name: str,
) -> None:
    """Validate one categorical probability distribution."""

    if (
        len(
            probabilities,
        )
        != expected_length
    ):
        msg = f"{name} probability count is {len(probabilities)}; expected {expected_length}."
        raise RuntimeError(msg)

    if not all(
        math.isfinite(
            probability,
        )
        for probability in probabilities
    ):
        msg = f"{name} probabilities contain non-finite values."
        raise RuntimeError(msg)

    if not all(0.0 <= probability <= 1.0 for probability in probabilities):
        msg = f"{name} probabilities are outside [0, 1]."
        raise RuntimeError(msg)

    probability_sum = sum(
        probabilities,
    )

    if not math.isclose(
        probability_sum,
        1.0,
        rel_tol=PROBABILITY_TOLERANCE,
        abs_tol=PROBABILITY_TOLERANCE,
    ):
        msg = f"{name} probabilities sum to {probability_sum}; expected 1.0."
        raise RuntimeError(msg)


def _validate_emotion(
    result: TacmeInferenceResult,
) -> None:
    """Validate the adapted emotion prediction."""

    prediction = result.emotion

    if prediction.class_index != EXPECTED_EMOTION_CLASS:
        msg = f"Emotion class index is {prediction.class_index}; expected {EXPECTED_EMOTION_CLASS}."
        raise RuntimeError(msg)

    if prediction.label != EXPECTED_EMOTION_LABEL:
        msg = f"Emotion label is {prediction.label!r}; expected {EXPECTED_EMOTION_LABEL!r}."
        raise RuntimeError(msg)

    if prediction.label != EMOTION_LABELS[prediction.class_index]:
        raise RuntimeError("Emotion label and class index are inconsistent.")

    _validate_probability_distribution(
        prediction.probabilities,
        expected_length=len(
            EMOTION_LABELS,
        ),
        name="Emotion",
    )

    if not math.isclose(
        prediction.confidence,
        prediction.probabilities[prediction.class_index],
        rel_tol=PROBABILITY_TOLERANCE,
        abs_tol=PROBABILITY_TOLERANCE,
    ):
        raise RuntimeError("Emotion confidence does not match the predicted-class probability.")


def _validate_personality(
    result: TacmeInferenceResult,
) -> None:
    """Validate the adapted personality prediction."""

    prediction = result.personality

    if prediction.labels != PERSONALITY_LABELS:
        msg = f"Unexpected personality labels: {prediction.labels!r}."
        raise RuntimeError(msg)

    if len(
        prediction.scores,
    ) != len(
        EXPECTED_PERSONALITY_SCORES,
    ):
        msg = f"Personality score count is {len(prediction.scores)}; expected {len(EXPECTED_PERSONALITY_SCORES)}."
        raise RuntimeError(msg)

    for label, actual, expected in zip(
        prediction.labels,
        prediction.scores,
        EXPECTED_PERSONALITY_SCORES,
        strict=True,
    ):
        if not math.isfinite(
            actual,
        ):
            msg = f"Personality score {label!r} is not finite."
            raise RuntimeError(msg)

        if not math.isclose(
            actual,
            expected,
            rel_tol=PROBABILITY_TOLERANCE,
            abs_tol=PROBABILITY_TOLERANCE,
        ):
            msg = f"Personality score {label!r} is {actual}; expected {expected}."
            raise RuntimeError(msg)


def _validate_ambivalence(
    result: TacmeInferenceResult,
) -> None:
    """Validate the adapted ambivalence prediction."""

    prediction = result.ambivalence

    if prediction.class_index != EXPECTED_AMBIVALENCE_CLASS:
        msg = f"Ambivalence class index is {prediction.class_index}; expected {EXPECTED_AMBIVALENCE_CLASS}."
        raise RuntimeError(msg)

    if prediction.label != EXPECTED_AMBIVALENCE_LABEL:
        msg = f"Ambivalence label is {prediction.label!r}; expected {EXPECTED_AMBIVALENCE_LABEL!r}."
        raise RuntimeError(msg)

    if prediction.label != AMBIVALENCE_LABELS[prediction.class_index]:
        raise RuntimeError("Ambivalence label and class index are inconsistent.")

    probabilities = (
        prediction.absence_probability,
        prediction.presence_probability,
    )

    _validate_probability_distribution(
        probabilities,
        expected_length=len(
            AMBIVALENCE_LABELS,
        ),
        name="Ambivalence",
    )

    if not math.isclose(
        prediction.confidence,
        probabilities[prediction.class_index],
        rel_tol=PROBABILITY_TOLERANCE,
        abs_tol=PROBABILITY_TOLERANCE,
    ):
        raise RuntimeError("Ambivalence confidence does not match the predicted-class probability.")


def _validate_missing_output_rejection() -> None:
    """Verify that adaptation rejects missing task outputs."""

    outputs = _create_test_outputs()

    outputs["ah_logits"] = None

    try:
        adapt_tacme_outputs(
            outputs,
        )

    except TacmeInferenceError:
        return

    raise RuntimeError("Inference adaptation accepted a missing ambivalence output.")


def _run_inference_check() -> None:
    """Run the TACME public inference-result smoke test."""

    result = adapt_tacme_outputs(
        _create_test_outputs(),
    )

    _validate_emotion(
        result,
    )

    _validate_personality(
        result,
    )

    _validate_ambivalence(
        result,
    )

    _validate_missing_output_rejection()

    print("TACME inference adaptation check passed.")
    print(
        "Emotion: "
        f"{result.emotion.label}, "
        f"class={result.emotion.class_index}, "
        f"confidence={result.emotion.confidence:.6f}"
    )
    print(
        "Personality: "
        + ", ".join(
            f"{label}={score:.6f}"
            for label, score in zip(
                result.personality.labels,
                result.personality.scores,
                strict=True,
            )
        )
    )
    print(
        "Ambivalence: "
        f"{result.ambivalence.label}, "
        f"class={result.ambivalence.class_index}, "
        f"confidence={result.ambivalence.confidence:.6f}"
    )
    print("Missing output rejection: passed")


def main() -> int:
    """Run the TACME inference adaptation smoke test."""

    try:
        _run_inference_check()

    except (
        RuntimeError,
        ValueError,
    ) as error:
        print(
            f"TACME inference adaptation check failed: {error}",
            file=sys.stderr,
        )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(),
    )
