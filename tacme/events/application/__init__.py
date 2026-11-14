"""
File: __init__.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Backward-compatible public interface for the TACME application event handlers.

License: MIT License
"""

from tacme.events.application.analysis import (
    handle_analysis_started as handle_analysis_started,
    handle_run_analysis as handle_run_analysis,
)
from tacme.events.application.controls import (
    handle_clear_application as handle_clear_application,
    handle_task_selection_change as handle_task_selection_change,
    handle_video_change as handle_video_change,
)
from tacme.events.application.results import (
    create_result_updates as create_result_updates,
    handle_localize_predictions_title as handle_localize_predictions_title,
)
from tacme.events.application.status import (
    create_analysis_completed_status as create_analysis_completed_status,
)

# Preserve the original module path used by external callers and serializers.
create_analysis_completed_status.__module__ = __name__
create_result_updates.__module__ = __name__
handle_analysis_started.__module__ = __name__
handle_clear_application.__module__ = __name__
handle_localize_predictions_title.__module__ = __name__
handle_run_analysis.__module__ = __name__
handle_task_selection_change.__module__ = __name__
handle_video_change.__module__ = __name__

__all__ = [
    "create_analysis_completed_status",
    "create_result_updates",
    "handle_analysis_started",
    "handle_clear_application",
    "handle_localize_predictions_title",
    "handle_run_analysis",
    "handle_task_selection_change",
    "handle_video_change",
]
