"""
File: __init__.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Backward-compatible attribution visualization API.

License: MIT License
"""

from tacme.visualization.attribution.constants import (
    ATTRIBUTION_AXIS as ATTRIBUTION_AXIS,
    ATTRIBUTION_HOVER_BACKGROUND as ATTRIBUTION_HOVER_BACKGROUND,
    ATTRIBUTION_NEGATIVE as ATTRIBUTION_NEGATIVE,
    ATTRIBUTION_NEUTRAL as ATTRIBUTION_NEUTRAL,
    ATTRIBUTION_POSITIVE as ATTRIBUTION_POSITIVE,
    ATTRIBUTION_VIEW_MEAN as ATTRIBUTION_VIEW_MEAN,
    ATTRIBUTION_VIEW_MEAN_STD as ATTRIBUTION_VIEW_MEAN_STD,
    ATTRIBUTION_VIEW_STD as ATTRIBUTION_VIEW_STD,
    ATTRIBUTION_VIEWS as ATTRIBUTION_VIEWS,
)
from tacme.visualization.attribution.figure import (
    create_feature_attribution_figure as create_feature_attribution_figure,
)
from tacme.visualization.attribution.rows import (
    _all_row_labels as _all_row_labels,
    _display_rows as _display_rows,
    _modality_labels as _modality_labels,
    _normalize_rows as _normalize_rows,
    _selected_row_indices as _selected_row_indices,
)
from tacme.visualization.attribution.summary import (
    _view_modality_magnitudes as _view_modality_magnitudes,
    create_feature_attribution_summary_html as create_feature_attribution_summary_html,
)

# Preserve the historical function paths for introspection and serialization.
for _legacy_function in (
    _selected_row_indices,
    _modality_labels,
    _all_row_labels,
    _display_rows,
    _normalize_rows,
    create_feature_attribution_figure,
    _view_modality_magnitudes,
    create_feature_attribution_summary_html,
):
    _legacy_function.__module__ = __name__
del _legacy_function

__all__ = [
    "ATTRIBUTION_VIEWS",
    "ATTRIBUTION_VIEW_MEAN",
    "ATTRIBUTION_VIEW_MEAN_STD",
    "ATTRIBUTION_VIEW_STD",
    "create_feature_attribution_figure",
    "create_feature_attribution_summary_html",
]
