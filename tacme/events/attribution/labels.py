"""
File: labels.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Localized attribution labels and UI selector options.

License: MIT License
"""

from collections.abc import Sequence

from tacme.explainability.attribution import (
    ATTRIBUTION_TASK_ORDER,
    FeatureAttributionBundle,
    FeatureAttributionTarget,
)
from tacme.inference import (
    AMBIVALENCE_LABELS,
    EMOTION_LABELS,
    PERSONALITY_LABELS,
)
from tacme.localization import get_localized_text, get_task_label
from tacme.visualization.attribution import (
    ATTRIBUTION_VIEW_MEAN,
    ATTRIBUTION_VIEW_MEAN_STD,
    ATTRIBUTION_VIEW_STD,
)


def _create_attribution_title(
    language_index: int,
) -> str:
    """Create the localized feature-attribution section title."""

    return "### " + get_localized_text(
        "Labels_ATTRIBUTION_SECTION_TITLE",
        language_index,
    )


def _create_attribution_note(
    _language_index: int,
) -> str:
    """Preserve the existing Gradio output contract without rendering a note."""

    return ""


def _task_choices(
    task_keys: Sequence[str],
    language_index: int,
) -> list[tuple[str, str]]:
    """Create localized task labels with canonical Gradio values."""

    return [
        (
            get_task_label(
                task_key,
                language_index,
            ),
            task_key,
        )
        for task_key in ATTRIBUTION_TASK_ORDER
        if task_key in task_keys
    ]


def _format_target_label(
    target: FeatureAttributionTarget,
    language_index: int,
) -> str:
    """Localize one target while preserving its canonical model identifier."""

    if target.task == "emotion":
        class_name = EMOTION_LABELS[target.target_index]
        key = f"Labels_SUMMARY_EMOTION_{class_name.upper()}"
        label = get_localized_text(key, language_index)
        return f"{label} · {target.score * 100.0:.2f}%"

    if target.task == "personality":
        class_name = PERSONALITY_LABELS[target.target_index]
        label = get_localized_text(
            f"Labels_PERSONALITY_{class_name}",
            language_index,
        )
        return f"{label} · {target.score:.4f}"

    class_name = AMBIVALENCE_LABELS[target.target_index]
    label = get_localized_text(
        f"Labels_AMBIVALENCE_{class_name.upper()}",
        language_index,
    )
    return f"{label} · {target.score * 100.0:.2f}%"


def _target_choices(
    bundle: FeatureAttributionBundle,
    task: str,
    language_index: int,
) -> list[tuple[str, str]]:
    """Create target selector choices for one TACME task."""

    return [
        (
            _format_target_label(
                target,
                language_index,
            ),
            target.target_id,
        )
        for target in bundle.targets_for_task(
            task,
        )
    ]


def _view_choices(
    language_index: int,
) -> list[tuple[str, str]]:
    """Create localized feature-attribution view choices."""

    return [
        (
            get_localized_text(
                "Labels_ATTRIBUTION_VIEW_MEAN_STD",
                language_index,
            ),
            ATTRIBUTION_VIEW_MEAN_STD,
        ),
        (
            get_localized_text(
                "Labels_ATTRIBUTION_VIEW_MEAN",
                language_index,
            ),
            ATTRIBUTION_VIEW_MEAN,
        ),
        (
            get_localized_text(
                "Labels_ATTRIBUTION_VIEW_STD",
                language_index,
            ),
            ATTRIBUTION_VIEW_STD,
        ),
    ]
