"""
File: summary.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: HTML summaries of relative attribution magnitudes.

License: MIT License
"""

from html import escape

from tacme.explainability.attribution import FeatureAttributionTarget
from tacme.localization import get_localized_text
from tacme.visualization.attribution.constants import (
    ATTRIBUTION_VIEW_MEAN,
    ATTRIBUTION_VIEW_STD,
)
from tacme.visualization.attribution.rows import _modality_labels


def _view_modality_magnitudes(
    target: FeatureAttributionTarget,
    view: str,
) -> tuple[float, ...]:
    """Calculate modality magnitudes from the currently displayed compact rows."""

    selected_statistics = (
        (0,)
        if view == ATTRIBUTION_VIEW_MEAN
        else (
            (1,)
            if view == ATTRIBUTION_VIEW_STD
            else (
                0,
                1,
            )
        )
    )

    magnitudes: list[float] = []

    for modality_index in range(
        4,
    ):
        selected_row_magnitudes = tuple(
            target.row_magnitudes[modality_index * 2 + statistic_index] for statistic_index in selected_statistics
        )

        magnitude = sum(
            selected_row_magnitudes,
        ) / len(
            selected_row_magnitudes,
        )

        magnitudes.append(
            magnitude,
        )

    total = sum(
        magnitudes,
    )

    if total <= 0.0:
        return tuple(0.0 for _ in magnitudes)

    return tuple(magnitude / total for magnitude in magnitudes)


def _display_percentage_tenths(
    magnitudes: tuple[float, ...],
) -> tuple[int, ...]:
    """Round visible percentages to tenths with an exact 100.0% total."""

    total = sum(magnitudes)

    if total <= 0.0:
        return tuple(0 for _ in magnitudes)

    scaled = tuple(magnitude / total * 1000.0 for magnitude in magnitudes)
    tenths = [int(value) for value in scaled]
    remaining = 1000 - sum(tenths)

    # Give leftover tenths to the largest fractional remainders.
    order = sorted(
        range(len(scaled)),
        key=lambda index: (scaled[index] - tenths[index], -index),
        reverse=True,
    )

    for index in order[:remaining]:
        tenths[index] += 1

    return tuple(tenths)


def create_feature_attribution_summary_html(
    target: FeatureAttributionTarget,
    view: str,
    language_index: int,
) -> str:
    """Render the relative modality-attribution summary beside the heatmap."""

    title = escape(
        get_localized_text(
            "Labels_ATTRIBUTION_RELATIVE_MAGNITUDE",
            language_index,
        )
    )

    modality_labels = _modality_labels(
        language_index,
    )
    magnitudes = _view_modality_magnitudes(
        target,
        view,
    )
    display_tenths = _display_percentage_tenths(magnitudes)

    rows = "".join(
        (
            '<div class="feature-attribution-summary-row">'
            '<div class="feature-attribution-summary-header">'
            f"<span>{escape(modality_label)}</span>"
            f"<strong>{tenths / 10.0:.1f}%</strong>"
            "</div>"
            '<div class="feature-attribution-summary-track">'
            '<span class="feature-attribution-summary-fill" '
            f'style="--tacme-attribution-share: {magnitude * 100.0:.3f}%;"></span>'
            "</div>"
            "</div>"
        )
        for modality_label, magnitude, tenths in zip(
            modality_labels,
            magnitudes,
            display_tenths,
            strict=True,
        )
    )

    return f"""
<section class="feature-attribution-summary">
    <div class="feature-attribution-summary-title">{title}</div>
    <div class="feature-attribution-summary-rows">
        {rows}
    </div>
</section>
""".strip()
