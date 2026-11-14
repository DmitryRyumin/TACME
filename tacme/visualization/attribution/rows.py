"""
File: rows.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Attribution row selection, normalization, and localized labels.

License: MIT License
"""

from tacme.explainability.attribution import FeatureAttributionTarget
from tacme.localization import get_localized_text
from tacme.visualization.attribution.constants import (
    ATTRIBUTION_VIEW_MEAN,
    ATTRIBUTION_VIEW_STD,
)


def _selected_row_indices(
    view: str,
) -> tuple[int, ...]:
    """Return row indices displayed for one attribution view."""

    if view == ATTRIBUTION_VIEW_MEAN:
        return (
            0,
            2,
            4,
            6,
        )

    if view == ATTRIBUTION_VIEW_STD:
        return (
            1,
            3,
            5,
            7,
        )

    return tuple(
        range(
            8,
        )
    )


def _modality_labels(
    language_index: int,
) -> tuple[str, ...]:
    """Return localized attribution modality labels."""

    return (
        get_localized_text(
            "Labels_ATTRIBUTION_LINGUISTIC",
            language_index,
        ),
        get_localized_text(
            "Labels_ATTRIBUTION_ACOUSTIC",
            language_index,
        ),
        get_localized_text(
            "Labels_ATTRIBUTION_BEHAVIORAL",
            language_index,
        ),
        get_localized_text(
            "Labels_ATTRIBUTION_VISUAL",
            language_index,
        ),
    )


def _all_row_labels(
    language_index: int,
) -> tuple[str, ...]:
    """Return localized Mean/STD labels for all four modalities."""

    mean_label = get_localized_text(
        "Labels_ATTRIBUTION_MEAN",
        language_index,
    )
    std_label = get_localized_text(
        "Labels_ATTRIBUTION_STD",
        language_index,
    )

    labels: list[str] = []

    for modality_label in _modality_labels(
        language_index,
    ):
        labels.extend(
            (
                f"{modality_label} · {mean_label}",
                f"{modality_label} · {std_label}",
            )
        )

    return tuple(
        labels,
    )


def _display_rows(
    target: FeatureAttributionTarget,
    view: str,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[int, ...],
]:
    """Return attribution rows and original row indices for one view."""

    row_indices = _selected_row_indices(
        view,
    )

    rows = tuple(target.values[index] for index in row_indices)

    return (
        rows,
        row_indices,
    )


def _normalize_rows(
    rows: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    """Apply one shared symmetric normalization to the displayed map."""

    max_abs = max(
        (abs(value) for row in rows for value in row),
        default=0.0,
    )

    if max_abs <= 0.0:
        return tuple(tuple(0.0 for _ in row) for row in rows)

    return tuple(tuple(value / max_abs for value in row) for row in rows)
