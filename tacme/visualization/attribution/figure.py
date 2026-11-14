"""
File: figure.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Interactive Plotly attribution heatmaps.

License: MIT License
"""

import plotly.graph_objects as go

from tacme.explainability.attribution import FeatureAttributionBundle, FeatureAttributionTarget
from tacme.localization import get_localized_text
from tacme.visualization.attribution.constants import (
    ATTRIBUTION_AXIS,
    ATTRIBUTION_HOVER_BACKGROUND,
    ATTRIBUTION_NEGATIVE,
    ATTRIBUTION_NEUTRAL,
    ATTRIBUTION_POSITIVE,
    ATTRIBUTION_VIEW_MEAN_STD,
    ATTRIBUTION_VIEWS,
)
from tacme.visualization.attribution.rows import (
    _all_row_labels,
    _display_rows,
    _normalize_rows,
)

_NEAR_ZERO_RELATIVE_THRESHOLD = 0.05


def _contribution_type(
    normalized_value: float,
    positive_label: str,
    negative_label: str,
    near_zero_label: str,
) -> str:
    """Classify only the displayed attribution, relative to its view maximum."""

    if abs(normalized_value) <= _NEAR_ZERO_RELATIVE_THRESHOLD:
        return near_zero_label

    return positive_label if normalized_value > 0.0 else negative_label


def _format_attribution_value(value: float) -> str:
    """Format a signed contribution without hiding very small nonzero values."""

    if value == 0.0:
        return "0.000000"

    if abs(value) < 0.00001:
        return f"{value:+.2e}"

    return f"{value:+.6f}"


def create_feature_attribution_figure(
    bundle: FeatureAttributionBundle,
    target: FeatureAttributionTarget,
    view: str,
    language_index: int,
) -> go.Figure:
    """Create the interactive Plotly heatmap for one TACME attribution target."""

    if view not in ATTRIBUTION_VIEWS:
        view = ATTRIBUTION_VIEW_MEAN_STD

    rows, row_indices = _display_rows(
        target,
        view,
    )
    normalized_rows = _normalize_rows(
        rows,
    )

    all_row_labels = _all_row_labels(
        language_index,
    )
    row_labels = tuple(all_row_labels[index] for index in row_indices)

    feature_bin_label = get_localized_text(
        "Labels_ATTRIBUTION_FEATURE_BIN",
        language_index,
    )
    attribution_label = get_localized_text(
        "Labels_ATTRIBUTION_VALUE",
        language_index,
    )
    estimated_label = get_localized_text(
        "Labels_ATTRIBUTION_ESTIMATED_VALUE",
        language_index,
    )
    of_label = get_localized_text(
        "Labels_ATTRIBUTION_OF",
        language_index,
    )
    feature_bin_hint = get_localized_text(
        "Labels_ATTRIBUTION_FEATURE_BINS_HINT",
        language_index,
    ).format(count=bundle.feature_bin_count)
    direction_label = get_localized_text(
        "Labels_ATTRIBUTION_DIRECTION",
        language_index,
    )
    supports_label = get_localized_text(
        "Labels_ATTRIBUTION_SUPPORTS",
        language_index,
    )
    opposes_label = get_localized_text(
        "Labels_ATTRIBUTION_OPPOSES",
        language_index,
    )
    neutral_label = get_localized_text(
        "Labels_ATTRIBUTION_NEUTRAL",
        language_index,
    )
    positive_direction = get_localized_text(
        "Labels_ATTRIBUTION_DIRECTION_POSITIVE",
        language_index,
    )
    negative_direction = get_localized_text(
        "Labels_ATTRIBUTION_DIRECTION_NEGATIVE",
        language_index,
    )
    zero_direction = get_localized_text(
        "Labels_ATTRIBUTION_DIRECTION_ZERO",
        language_index,
    )

    custom_data = [
        [
            [
                raw_value,
                _contribution_type(
                    normalized_value,
                    positive_direction,
                    negative_direction,
                    zero_direction,
                ),
                _format_attribution_value(raw_value),
            ]
            for raw_value, normalized_value in zip(
                row,
                normalized_row,
                strict=True,
            )
        ]
        for row, normalized_row in zip(
            rows,
            normalized_rows,
            strict=True,
        )
    ]

    figure = go.Figure(
        data=[
            go.Heatmap(
                z=normalized_rows,
                x=tuple(
                    range(
                        1,
                        bundle.feature_bin_count + 1,
                    )
                ),
                y=row_labels,
                customdata=custom_data,
                zmin=-1.0,
                zmax=1.0,
                zmid=0.0,
                colorscale=[
                    [
                        0.0,
                        ATTRIBUTION_NEGATIVE,
                    ],
                    [
                        0.22,
                        "#93C5FD",
                    ],
                    [
                        0.5,
                        ATTRIBUTION_NEUTRAL,
                    ],
                    [
                        0.78,
                        "#FDBA74",
                    ],
                    [
                        1.0,
                        ATTRIBUTION_POSITIVE,
                    ],
                ],
                colorbar={
                    "title": {
                        "text": attribution_label,
                        "side": "right",
                        "font": {
                            "color": ATTRIBUTION_AXIS,
                            "size": 11,
                        },
                    },
                    "tickmode": "array",
                    "tickvals": [
                        -1.0,
                        0.0,
                        1.0,
                    ],
                    "ticktext": [
                        opposes_label,
                        neutral_label,
                        supports_label,
                    ],
                    "tickfont": {
                        "color": ATTRIBUTION_AXIS,
                        "size": 10,
                    },
                    "thickness": 11,
                    "len": 0.78,
                    "outlinewidth": 0,
                    "xpad": 8,
                },
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"<b>{feature_bin_label} %{{x}} {of_label} {bundle.feature_bin_count}</b><br>"
                    f"<b>{estimated_label}</b>: %{{customdata[2]}}<br>"
                    f"<b>{direction_label}</b>: %{{customdata[1]}}"
                    "<extra></extra>"
                ),
                xgap=2,
                ygap=2,
            )
        ]
    )

    figure.update_layout(
        template="none",
        autosize=True,
        meta={"tacme_feature_bins_hint": feature_bin_hint},
        height=(350 if view == ATTRIBUTION_VIEW_MEAN_STD else 286),
        margin={
            "l": 126,
            "r": 196,
            "t": 8,
            "b": 46,
        },
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font={
            "color": ATTRIBUTION_AXIS,
            "size": 12,
        },
        hoverlabel={
            "bgcolor": ATTRIBUTION_HOVER_BACKGROUND,
            "bordercolor": ATTRIBUTION_POSITIVE,
            "font": {
                "color": "#FAFAFA",
                "size": 12,
            },
        },
        xaxis={
            "title": {
                "text": get_localized_text(
                    "Labels_ATTRIBUTION_FEATURE_BINS",
                    language_index,
                ),
                "standoff": 10,
            },
            "showgrid": False,
            "zeroline": False,
            "fixedrange": True,
            "tickfont": {
                "color": ATTRIBUTION_AXIS,
                "size": 11,
            },
            "title_font": {
                "color": ATTRIBUTION_AXIS,
                "size": 11,
            },
            "tickmode": "array",
            "tickvals": [
                1,
                8,
                16,
                24,
                bundle.feature_bin_count,
            ],
            "ticktext": [
                "1",
                "8",
                "16",
                "24",
                str(
                    bundle.feature_bin_count,
                ),
            ],
            "side": "bottom",
        },
        yaxis={
            "autorange": "reversed",
            "showgrid": False,
            "zeroline": False,
            "fixedrange": True,
            "tickfont": {
                "color": ATTRIBUTION_AXIS,
                "size": 11,
            },
        },
        uirevision=f"tacme-feature-attribution-{target.target_id}-{view}",
    )

    return figure
