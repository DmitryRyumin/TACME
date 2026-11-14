"""
File: __init__.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Backward-compatible feature-attribution event-handler API.

License: MIT License
"""

# Re-export all historical public names for event_handlers.py and other callers.
from tacme.events.attribution.client import (
    FEATURE_ATTRIBUTION_JS as FEATURE_ATTRIBUTION_JS,
    FEATURE_ATTRIBUTION_RESIZE_JS as FEATURE_ATTRIBUTION_RESIZE_JS,
)
from tacme.events.attribution.handlers import (
    handle_clear_feature_attribution as handle_clear_feature_attribution,
    handle_feature_attribution_target_change as handle_feature_attribution_target_change,
    handle_feature_attribution_task_change as handle_feature_attribution_task_change,
    handle_feature_attribution_view_change as handle_feature_attribution_view_change,
    handle_render_feature_attribution as handle_render_feature_attribution,
)
from tacme.events.attribution.labels import (
    _create_attribution_note as _create_attribution_note,
    _create_attribution_title as _create_attribution_title,
    _format_target_label as _format_target_label,
    _target_choices as _target_choices,
    _task_choices as _task_choices,
    _view_choices as _view_choices,
)
from tacme.events.attribution.selection import (
    _resolve_target as _resolve_target,
    _resolve_task as _resolve_task,
    _resolve_view as _resolve_view,
    _selected_task_keys as _selected_task_keys,
)

# Preserve the historical function path for serialization and introspection.
for _legacy_handler in (
    handle_clear_feature_attribution,
    handle_render_feature_attribution,
    handle_feature_attribution_task_change,
    handle_feature_attribution_target_change,
    handle_feature_attribution_view_change,
):
    _legacy_handler.__module__ = __name__
del _legacy_handler

__all__ = [
    "FEATURE_ATTRIBUTION_JS",
    "FEATURE_ATTRIBUTION_RESIZE_JS",
    "handle_clear_feature_attribution",
    "handle_feature_attribution_target_change",
    "handle_feature_attribution_task_change",
    "handle_feature_attribution_view_change",
    "handle_render_feature_attribution",
]
