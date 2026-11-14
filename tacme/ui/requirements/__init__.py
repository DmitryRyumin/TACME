"""TACME Requirements tab and its backwards-compatible public API."""

from tacme.ui.requirements.dependencies import (
    LOGGER,
    PYPI_JSON_URL_TEMPLATE,
    PYPI_TIMEOUT_SECONDS,
    PYPROJECT_PATH,
    _fetch_latest_pypi_version,
    _get_installed_version,
    _read_project_metadata,
    get_runtime_dependencies,
)
from tacme.ui.requirements.models import (
    RequirementsSessionState,
    RequirementsTabComponents,
    RuntimeDependency,
)
from tacme.ui.requirements.rendering import (
    PACKAGE_DISPLAY_NAMES,
    PACKAGE_PURPOSE_KEYS,
    PYPI_HOME_URL,
    PYPROJECT_GITHUB_URL,
    STATUS_CSS_CLASSES,
    STATUS_LOCALIZATION_KEYS,
    _create_last_checked_html,
    _create_loader_html,
    _create_package_cell_html,
    _create_pypi_link_html,
    _create_pyproject_link_html,
    _create_runtime_summary_html,
    _create_status_html,
    _create_version_badge_html,
    _get_dependency_purpose,
    _get_package_display_name,
    _resolve_dependency_status,
    create_initial_requirements_content_html,
    create_requirements_content_html,
    create_requirements_content_html_from_state,
    render_requirements_session_state,
)
from tacme.ui.requirements.session import (
    PYPI_MAX_WORKERS,
    _create_pending_session_state,
    stream_requirements_session_state,
)
from tacme.ui.requirements.tab import create_requirements_tab

__all__ = [
    "RequirementsSessionState",
    "RequirementsTabComponents",
    "RuntimeDependency",
    "create_initial_requirements_content_html",
    "create_requirements_content_html",
    "create_requirements_content_html_from_state",
    "create_requirements_tab",
    "get_runtime_dependencies",
    "render_requirements_session_state",
    "stream_requirements_session_state",
]
