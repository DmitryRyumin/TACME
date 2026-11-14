"""
File: __init__.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Backward-compatible public analysis-summary event API.

License: MIT License
"""

from tacme.events.analysis_summary.client import ANALYSIS_SUMMARY_JS as ANALYSIS_SUMMARY_JS
from tacme.events.analysis_summary.handlers import (
    handle_clear_analysis_summary as handle_clear_analysis_summary,
    handle_render_analysis_summary as handle_render_analysis_summary,
)
from tacme.events.analysis_summary.rendering import create_analysis_summary_html as create_analysis_summary_html

# Preserve the historical public function paths for serialization and introspection.
for _legacy_function in (
    create_analysis_summary_html,
    handle_render_analysis_summary,
    handle_clear_analysis_summary,
):
    _legacy_function.__module__ = __name__
del _legacy_function

__all__ = [
    "ANALYSIS_SUMMARY_JS",
    "create_analysis_summary_html",
    "handle_clear_analysis_summary",
    "handle_render_analysis_summary",
]
