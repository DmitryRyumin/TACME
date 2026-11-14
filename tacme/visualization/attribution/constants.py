"""
File: constants.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Styling and view constants for attribution visualizations.

License: MIT License
"""

from typing import Final

ATTRIBUTION_NEGATIVE: Final = "#3B82F6"
ATTRIBUTION_NEUTRAL: Final = "#F4F4F5"
ATTRIBUTION_POSITIVE: Final = "#F97316"
ATTRIBUTION_AXIS: Final = "#52525B"
ATTRIBUTION_HOVER_BACKGROUND: Final = "#18181B"

ATTRIBUTION_VIEW_MEAN_STD: Final = "mean_std"
ATTRIBUTION_VIEW_MEAN: Final = "mean"
ATTRIBUTION_VIEW_STD: Final = "std"

ATTRIBUTION_VIEWS: Final[tuple[str, ...]] = (
    ATTRIBUTION_VIEW_MEAN_STD,
    ATTRIBUTION_VIEW_MEAN,
    ATTRIBUTION_VIEW_STD,
)
