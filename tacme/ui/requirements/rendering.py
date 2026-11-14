"""
File: rendering.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: TACME Requirements tab implementation.

License: MIT License
"""

from datetime import UTC, datetime
from html import escape
import platform
from typing import Final
from urllib.parse import quote

from packaging.version import InvalidVersion, Version

from tacme.localization import get_language_index, get_localized_text
from tacme.ui.requirements.dependencies import get_runtime_dependencies
from tacme.ui.requirements.models import RequirementsSessionState, RuntimeDependency
from tacme.ui.requirements.session import _create_pending_session_state

PYPROJECT_GITHUB_URL: Final = "https://github.com/DmitryRyumin/TACME/blob/main/pyproject.toml"


PYPI_HOME_URL: Final = "https://pypi.org/"


PACKAGE_DISPLAY_NAMES: Final[dict[str, str]] = {
    "gradio": "Gradio",
    "packaging": "packaging",
    "pillow": "Pillow",
    "psutil": "psutil",
    "safetensors": "Safetensors",
    "torch": "PyTorch",
    "transformers": "Transformers",
    "torchvision": "TorchVision",
    "torchcodec": "TorchCodec",
    "qwen-vl-utils": "Qwen VL Utils",
}


PACKAGE_PURPOSE_KEYS: Final[dict[str, str]] = {
    "gradio": "Requirements_PURPOSE_GRADIO",
    "packaging": "Requirements_PURPOSE_PACKAGING",
    "pillow": "Requirements_PURPOSE_PILLOW",
    "psutil": "Requirements_PURPOSE_PSUTIL",
    "safetensors": "Requirements_PURPOSE_SAFETENSORS",
    "torch": "Requirements_PURPOSE_TORCH",
    "torchvision": "Requirements_PURPOSE_TORCHVISION",
    "torchcodec": "Requirements_PURPOSE_TORCHCODEC",
    "transformers": "Requirements_PURPOSE_TRANSFORMERS",
    "qwen-vl-utils": "Requirements_PURPOSE_QWEN_VL_UTILS",
}


STATUS_LOCALIZATION_KEYS: Final[dict[str, str]] = {
    "current": "Requirements_STATUS_CURRENT",
    "update": "Requirements_STATUS_UPDATE_AVAILABLE",
    "constrained": "Requirements_STATUS_CONSTRAINED",
    "missing": "Requirements_STATUS_MISSING",
    "incompatible": "Requirements_STATUS_INCOMPATIBLE",
    "unavailable": "Requirements_STATUS_UNAVAILABLE",
}


STATUS_CSS_CLASSES: Final[dict[str, str]] = {
    "current": "is-current",
    "update": "is-update",
    "constrained": "is-constrained",
    "missing": "is-error",
    "incompatible": "is-error",
    "unavailable": "is-unavailable",
}


def _get_package_display_name(
    dependency: RuntimeDependency,
) -> str:
    """Create a display name including configured extras."""

    display_name = PACKAGE_DISPLAY_NAMES.get(
        dependency.canonical_name,
        dependency.requirement.name,
    )

    if not dependency.requirement.extras:
        return display_name

    extras = ", ".join(
        sorted(
            dependency.requirement.extras,
        )
    )

    return f"{display_name} [{extras}]"


def _get_dependency_purpose(
    dependency: RuntimeDependency,
    language_index: int,
) -> str:
    """Return the localized TACME dependency purpose."""

    localization_key = PACKAGE_PURPOSE_KEYS.get(
        dependency.canonical_name,
    )

    if localization_key is None:
        return get_localized_text(
            "Requirements_PURPOSE_GENERIC",
            language_index,
        )

    return get_localized_text(
        localization_key,
        language_index,
    )


def _resolve_dependency_status(
    dependency: RuntimeDependency,
    latest_version: str | None,
    *,
    pending: bool,
) -> str | None:
    """Resolve the dependency status displayed in the table."""

    installed_version = dependency.installed_version

    if installed_version is None:
        return "missing"

    try:
        installed = Version(
            installed_version,
        )

    except InvalidVersion:
        return "incompatible"

    specifier = dependency.requirement.specifier

    if specifier and not specifier.contains(
        installed,
        prereleases=True,
    ):
        return "incompatible"

    if pending:
        return None

    if latest_version is None:
        return "unavailable"

    try:
        latest = Version(
            latest_version,
        )

    except InvalidVersion:
        return "unavailable"

    if installed >= latest:
        return "current"

    if not specifier or specifier.contains(
        latest,
        prereleases=True,
    ):
        return "update"

    return "constrained"


def _create_loader_html(
    language_index: int,
) -> str:
    """Create the neutral loader used during PyPI lookup."""

    aria_label = escape(
        get_localized_text(
            "Requirements_CHECKING_LATEST",
            language_index,
        ),
        quote=True,
    )

    return f'<span class="requirements-version-loader" role="status" aria-label="{aria_label}"></span>'


def _create_version_badge_html(
    version: str,
) -> str:
    """Create a compact version badge."""

    return f'<span class="requirements-version-badge">{escape(version)}</span>'


def _create_status_html(
    status: str | None,
    language_index: int,
) -> str:
    """Create a localized dependency status badge."""

    if status is None:
        return '<span class="requirements-status-pending" aria-hidden="true"></span>'

    localization_key = STATUS_LOCALIZATION_KEYS[status]

    css_class = STATUS_CSS_CLASSES[status]

    label = get_localized_text(
        localization_key,
        language_index,
    )

    return (
        "<span "
        'class="requirements-status-badge '
        f'{css_class}">'
        "<span "
        'class="requirements-status-dot" '
        'aria-hidden="true">'
        "</span>"
        f"{escape(label)}"
        "</span>"
    )


def _create_package_cell_html(
    dependency: RuntimeDependency,
) -> str:
    """Create the linked package-name cell."""

    package_name = _get_package_display_name(
        dependency,
    )

    project_url = f"https://pypi.org/project/{quote(dependency.requirement.name, safe='')}/"

    return (
        "<a "
        'class="requirements-package-link" '
        f'href="{escape(project_url, quote=True)}" '
        'target="_blank" '
        'rel="noopener noreferrer">'
        f"{escape(package_name)}"
        "</a>"
    )


def _create_pyproject_link_html() -> str:
    """Create the GitHub pyproject.toml source link."""

    return (
        "<a "
        'class="requirements-source-link" '
        f'href="{escape(PYPROJECT_GITHUB_URL, quote=True)}" '
        'target="_blank" '
        'rel="noopener noreferrer">'
        "<span>pyproject.toml</span>"
        "<span "
        'class="requirements-source-link-arrow" '
        'aria-hidden="true">'
        "↗"
        "</span>"
        "</a>"
    )


def _create_pypi_link_html() -> str:
    """Create the primary PyPI source link."""

    return (
        "<a "
        'class="requirements-summary-primary-link" '
        f'href="{escape(PYPI_HOME_URL, quote=True)}" '
        'target="_blank" '
        'rel="noopener noreferrer">'
        "<span>PyPI</span>"
        "<span "
        'class="requirements-summary-primary-link-arrow" '
        'aria-hidden="true">'
        "↗"
        "</span>"
        "</a>"
    )


def _create_runtime_summary_html(
    requires_python: str,
    dependency_count: int,
    language_index: int,
) -> str:
    """Create compact runtime summary cards."""

    python_label = get_localized_text(
        "Requirements_SUMMARY_PYTHON",
        language_index,
    )

    dependencies_label = get_localized_text(
        "Requirements_SUMMARY_DEPENDENCIES",
        language_index,
    )

    source_label = get_localized_text(
        "Requirements_SUMMARY_SOURCE",
        language_index,
    )

    source_meta = get_localized_text(
        "Requirements_SUMMARY_SOURCE_META",
        language_index,
    )

    required_label = get_localized_text(
        "Requirements_REQUIRED",
        language_index,
    )

    installed_python = platform.python_version()

    pyproject_link = _create_pyproject_link_html()
    pypi_link = _create_pypi_link_html()

    return f"""
    <div class="requirements-summary-grid">
        <div class="requirements-summary-card">
            <span class="requirements-summary-label">
                {escape(python_label)}
            </span>

            <strong class="requirements-summary-value">
                {escape(installed_python)}
            </strong>

            <span class="requirements-summary-meta">
                {escape(required_label)}
                {escape(requires_python or "—")}
            </span>
        </div>

        <div class="requirements-summary-card">
            <span class="requirements-summary-label">
                {escape(dependencies_label)}
            </span>

            <strong class="requirements-summary-value">
                {dependency_count}
            </strong>

            <span class="requirements-summary-meta">
                {pyproject_link}
            </span>
        </div>

        <div class="requirements-summary-card">
            <span class="requirements-summary-label">
                {escape(source_label)}
            </span>

            <strong
                class="
                    requirements-summary-value
                    requirements-summary-source
                "
            >
                {pypi_link}
            </strong>

            <span class="requirements-summary-meta">
                {pyproject_link}

                <span class="requirements-source-separator">
                    +
                </span>

                <span>
                    {escape(source_meta)}
                </span>
            </span>
        </div>
    </div>
    """


def _create_last_checked_html(
    language_index: int,
    checked_at: datetime | None,
) -> str:
    """Create the latest PyPI-check timestamp."""

    if checked_at is None:
        return ""

    label = get_localized_text(
        "Requirements_LAST_CHECKED",
        language_index,
    )

    checked_time = checked_at.astimezone(
        UTC,
    ).strftime(
        "%H:%M",
    )

    return (
        '<span class="requirements-last-checked">'
        f"{escape(label)}"
        '<span aria-hidden="true"> · </span>'
        "<strong>"
        f"{checked_time} UTC"
        "</strong>"
        "</span>"
    )


def create_requirements_content_html(
    language_index: int,
    session_state: RequirementsSessionState | None,
    *,
    animate_resolved: bool = False,
) -> str:
    """Render the complete TACME Requirements table."""

    requires_python, dependencies = get_runtime_dependencies()

    resolved_state = session_state or _create_pending_session_state()

    latest_versions = dict(
        resolved_state.latest_versions,
    )

    pending_names = resolved_state.pending_names

    package_label = get_localized_text(
        "Requirements_COLUMN_PACKAGE",
        language_index,
    )

    purpose_label = get_localized_text(
        "Requirements_COLUMN_PURPOSE",
        language_index,
    )

    requirement_label = get_localized_text(
        "Requirements_COLUMN_REQUIREMENT",
        language_index,
    )

    installed_label = get_localized_text(
        "Requirements_COLUMN_INSTALLED",
        language_index,
    )

    latest_label = get_localized_text(
        "Requirements_COLUMN_LATEST",
        language_index,
    )

    status_label = get_localized_text(
        "Requirements_COLUMN_STATUS",
        language_index,
    )

    unavailable_label = get_localized_text(
        "Requirements_VALUE_UNAVAILABLE",
        language_index,
    )

    rows: list[str] = []

    for dependency in dependencies:
        canonical_name = dependency.canonical_name

        pending = canonical_name in pending_names

        latest_version = latest_versions.get(
            canonical_name,
        )

        status = _resolve_dependency_status(
            dependency,
            latest_version,
            pending=pending,
        )

        installed_html = (
            _create_version_badge_html(
                dependency.installed_version,
            )
            if dependency.installed_version is not None
            else (f'<span class="requirements-value-unavailable">{escape(unavailable_label)}</span>')
        )

        if pending:
            latest_html = _create_loader_html(
                language_index,
            )

        elif latest_version is not None:
            latest_html = _create_version_badge_html(
                latest_version,
            )

        else:
            latest_html = f'<span class="requirements-value-unavailable">{escape(unavailable_label)}</span>'

        row_class = " is-resolved" if (animate_resolved and canonical_name == resolved_state.resolved_name) else ""

        rows.append(
            """
            <tr class="requirements-dependency-row{row_class}">
                <td class="requirements-package-cell">
                    {package}
                </td>

                <td class="requirements-purpose-cell">
                    {purpose}
                </td>

                <td>
                    <code class="requirements-specifier">
                        {requirement}
                    </code>
                </td>

                <td class="requirements-version-cell">
                    {installed}
                </td>

                <td
                    class="
                        requirements-version-cell
                        requirements-latest-cell
                    "
                >
                    {latest}
                </td>

                <td class="requirements-status-cell">
                    {status}
                </td>
            </tr>
            """.format(
                row_class=row_class,
                package=_create_package_cell_html(
                    dependency,
                ),
                purpose=escape(
                    _get_dependency_purpose(
                        dependency,
                        language_index,
                    )
                ),
                requirement=escape(
                    dependency.raw_requirement,
                ),
                installed=installed_html,
                latest=latest_html,
                status=_create_status_html(
                    status,
                    language_index,
                ),
            )
        )

    footer = get_localized_text(
        "Requirements_FOOTER",
        language_index,
    )

    last_checked_html = (
        _create_last_checked_html(
            language_index,
            resolved_state.checked_at,
        )
        if resolved_state.completed
        else ""
    )

    return f"""
    <section class="requirements-shell">
        {
        _create_runtime_summary_html(
            requires_python,
            len(dependencies),
            language_index,
        )
    }

        <div class="requirements-table-wrapper">
            <table class="requirements-table">
                <thead>
                    <tr>
                        <th>
                            {escape(package_label)}
                        </th>

                        <th>
                            {escape(purpose_label)}
                        </th>

                        <th>
                            {escape(requirement_label)}
                        </th>

                        <th>
                            {escape(installed_label)}
                        </th>

                        <th>
                            {escape(latest_label)}
                        </th>

                        <th>
                            {escape(status_label)}
                        </th>
                    </tr>
                </thead>

                <tbody>
                    {"".join(rows)}
                </tbody>
            </table>
        </div>

        <div class="requirements-footer-row">
            <p class="requirements-footer">
                {escape(footer)}
            </p>

            {last_checked_html}
        </div>
    </section>
    """


def create_initial_requirements_content_html(
    language_index: int,
) -> str:
    """Create the initial pending table without network requests."""

    return create_requirements_content_html(
        language_index,
        None,
        animate_resolved=False,
    )


def create_requirements_content_html_from_state(
    language_index: int,
    session_state: RequirementsSessionState | None,
) -> str:
    """Render localized content without checking PyPI again."""

    return create_requirements_content_html(
        language_index,
        session_state,
        animate_resolved=False,
    )


def render_requirements_session_state(
    language: str,
    session_state: RequirementsSessionState | None,
) -> str:
    """Render one streamed state update in the current language."""

    language_index = get_language_index(
        language,
    )

    return create_requirements_content_html(
        language_index,
        session_state,
        animate_resolved=True,
    )
