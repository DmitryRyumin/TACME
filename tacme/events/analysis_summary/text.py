"""
File: text.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Localized, deterministic emotion, personality, and ambivalence summary text.

License: MIT License
"""

from dataclasses import dataclass
from typing import Final

from tacme.inference import TacmeInferenceResult
from tacme.localization import get_localized_text

_PERSONALITY_CONFIG_FIELDS: Final[tuple[str, ...]] = (
    "Labels_PERSONALITY_OPE",
    "Labels_PERSONALITY_CON",
    "Labels_PERSONALITY_EXT",
    "Labels_PERSONALITY_AGR",
    "Labels_PERSONALITY_NNEU",
)

_EMOTION_CONFIG_FIELDS: Final[tuple[str, ...]] = (
    "Labels_SUMMARY_EMOTION_NEUTRAL",
    "Labels_SUMMARY_EMOTION_ANGER",
    "Labels_SUMMARY_EMOTION_DISGUST",
    "Labels_SUMMARY_EMOTION_FEAR",
    "Labels_SUMMARY_EMOTION_HAPPINESS",
    "Labels_SUMMARY_EMOTION_SADNESS",
    "Labels_SUMMARY_EMOTION_SURPRISE",
)

_AMBIVALENCE_PHRASE_CONFIG_FIELDS: Final[tuple[str, ...]] = (
    "Labels_SUMMARY_AMBIVALENCE_ABSENCE_PHRASE",
    "Labels_SUMMARY_AMBIVALENCE_PRESENCE_PHRASE",
)

_AMBIVALENCE_SHORT_CONFIG_FIELDS: Final[tuple[str, ...]] = (
    "Labels_SUMMARY_AMBIVALENCE_ABSENCE_SHORT",
    "Labels_SUMMARY_AMBIVALENCE_PRESENCE_SHORT",
)

_AMBIVALENCE_GENITIVE_CONFIG_FIELDS: Final[tuple[str, ...]] = (
    "Labels_SUMMARY_AMBIVALENCE_ABSENCE_GENITIVE",
    "Labels_SUMMARY_AMBIVALENCE_PRESENCE_GENITIVE",
)


def _localized_personality_name(
    trait_index: int,
    language_index: int,
) -> str:
    """Return the localized long name of one apparent personality trait."""

    return get_localized_text(
        _PERSONALITY_CONFIG_FIELDS[trait_index],
        language_index,
    )


def _localized_emotion_label(
    class_index: int,
    language_index: int,
) -> str:
    """Return one localized emotion class label."""

    return get_localized_text(
        _EMOTION_CONFIG_FIELDS[class_index],
        language_index,
    )


def _localized_ambivalence_phrase(
    class_index: int,
    language_index: int,
) -> str:
    """Return a localized grammatical phrase for one ambivalence class."""

    return get_localized_text(
        _AMBIVALENCE_PHRASE_CONFIG_FIELDS[class_index],
        language_index,
    )


@dataclass(frozen=True, slots=True)
class _LocalizedNumber:
    """Keep fixed decimal precision and localize the decimal separator."""

    value: float
    language_index: int

    def __format__(self, format_spec: str) -> str:
        formatted = format(self.value, format_spec)
        return formatted.replace(".", ",") if self.language_index == 1 else formatted


def _format_template(
    config_field: str,
    language_index: int,
    **values: object,
) -> str:
    """Format one localized deterministic summary template."""

    template = get_localized_text(
        config_field,
        language_index,
    )
    localized_values = {
        key: _LocalizedNumber(value, language_index) if isinstance(value, float) else value
        for key, value in values.items()
    }

    return template.format(**localized_values)


def _create_emotion_text(
    result: TacmeInferenceResult,
    language_index: int,
) -> tuple[str, str]:
    """Create the detailed and compact emotion-summary sentences."""

    prediction = result.emotion
    ranked_indices = sorted(
        range(
            len(
                prediction.probabilities,
            )
        ),
        key=lambda index: prediction.probabilities[index],
        reverse=True,
    )

    primary_index = ranked_indices[0]
    secondary_index = ranked_indices[1]
    primary_label = _localized_emotion_label(
        primary_index,
        language_index,
    )
    secondary_label = _localized_emotion_label(
        secondary_index,
        language_index,
    )
    primary_probability = prediction.probabilities[primary_index] * 100.0
    secondary_probability = prediction.probabilities[secondary_index] * 100.0

    detailed = _format_template(
        "Texts_SUMMARY_EMOTION_TEXT",
        language_index,
        primary_label=primary_label,
        primary_probability=primary_probability,
        secondary_label=secondary_label,
        secondary_probability=secondary_probability,
    )

    compact = _format_template(
        "Texts_SUMMARY_FINAL_EMOTION_TEXT",
        language_index,
        primary_label_lower=primary_label.lower(),
        primary_probability=primary_probability,
    )

    return (
        detailed,
        compact,
    )


def _create_personality_text(
    result: TacmeInferenceResult,
    language_index: int,
) -> tuple[str, str]:
    """Create deterministic personality-summary sentences without score thresholding."""

    scores = result.personality.scores
    ranked_indices = sorted(
        range(
            len(
                scores,
            )
        ),
        key=lambda index: scores[index],
        reverse=True,
    )

    first_index, second_index, third_index = ranked_indices[:3]
    lowest_index = ranked_indices[-1]

    values = {
        "first": _localized_personality_name(
            first_index,
            language_index,
        ).lower(),
        "first_score": scores[first_index],
        "second": _localized_personality_name(
            second_index,
            language_index,
        ).lower(),
        "second_score": scores[second_index],
        "third": _localized_personality_name(
            third_index,
            language_index,
        ).lower(),
        "third_score": scores[third_index],
        "lowest": _localized_personality_name(
            lowest_index,
            language_index,
        ).lower(),
        "lowest_score": scores[lowest_index],
    }

    detailed = _format_template(
        "Texts_SUMMARY_PERSONALITY_TEXT",
        language_index,
        **values,
    )

    compact = _format_template(
        "Texts_SUMMARY_FINAL_PERSONALITY_TEXT",
        language_index,
        **values,
    )

    return (
        detailed,
        compact,
    )


def _create_ambivalence_text(
    result: TacmeInferenceResult,
    language_index: int,
) -> tuple[str, str]:
    """Create the detailed and compact ambivalence-summary sentences."""

    prediction = result.ambivalence
    primary_index = prediction.class_index
    secondary_index = 1 - primary_index

    probabilities = (
        prediction.absence_probability,
        prediction.presence_probability,
    )

    primary_phrase = _localized_ambivalence_phrase(
        primary_index,
        language_index,
    )
    secondary_short_phrase = get_localized_text(
        _AMBIVALENCE_SHORT_CONFIG_FIELDS[secondary_index],
        language_index,
    )
    primary_probability_phrase = get_localized_text(
        _AMBIVALENCE_GENITIVE_CONFIG_FIELDS[primary_index],
        language_index,
    )
    primary_probability = probabilities[primary_index] * 100.0
    secondary_probability = probabilities[secondary_index] * 100.0

    detailed = _format_template(
        "Texts_SUMMARY_AMBIVALENCE_TEXT",
        language_index,
        primary_phrase=primary_phrase,
        primary_probability=primary_probability,
        secondary_short_phrase=secondary_short_phrase,
        secondary_probability=secondary_probability,
    )

    compact = _format_template(
        "Texts_SUMMARY_FINAL_AMBIVALENCE_TEXT",
        language_index,
        primary_probability_phrase=primary_probability_phrase,
        primary_probability=primary_probability,
    )

    return (
        detailed,
        compact,
    )
