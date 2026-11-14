"""README generator implementation for the TACME project.

Original authors: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin.
License: MIT License.
"""

from __future__ import annotations

import re

from scripts.readme.constants import (
    COMPATIBLE_VERSION_PATTERN,
    EXACT_VERSION_PATTERN,
    MINIMUM_VERSION_PATTERN,
    NODE_VERSION_PATH,
    NPM_PACKAGE_MANAGER_PATTERN,
    PYTHON_DEPENDENCY_PATTERN,
    Config,
    DependencyMap,
)
from scripts.readme.sources import (
    _read_text_value,
)


def _normalize_package_name(
    name: str,
) -> str:
    """Normalize a Python package name."""

    return re.sub(
        r"[-_.]+",
        "-",
        name.strip().lower(),
    )


def _parse_python_dependency(
    dependency: str,
) -> tuple[str, str]:
    """Split a Python dependency into name and version specification."""

    match = PYTHON_DEPENDENCY_PATTERN.match(
        dependency.strip(),
    )

    if match is None:
        return "", ""

    name = _normalize_package_name(
        match.group(1),
    )

    specification = match.group(
        2,
    ).strip()

    if ";" in specification:
        specification = specification.partition(
            ";",
        )[0].strip()

    return name, specification


def _add_python_dependencies(
    destination: DependencyMap,
    values: object,
) -> None:
    """Add dependency specifications from a list."""

    if not isinstance(
        values,
        list,
    ):
        return

    for dependency in values:
        if not isinstance(
            dependency,
            str,
        ):
            continue

        name, specification = _parse_python_dependency(
            dependency,
        )

        if name:
            destination[name] = specification


def _collect_python_dependencies(
    pyproject: Config,
) -> DependencyMap:
    """Collect runtime and development Python dependencies."""

    dependencies: DependencyMap = {}

    project = pyproject.get(
        "project",
    )

    if isinstance(
        project,
        dict,
    ):
        _add_python_dependencies(
            dependencies,
            project.get(
                "dependencies",
            ),
        )

        optional_dependencies = project.get(
            "optional-dependencies",
        )

        if isinstance(
            optional_dependencies,
            dict,
        ):
            for group in optional_dependencies.values():
                _add_python_dependencies(
                    dependencies,
                    group,
                )

    dependency_groups = pyproject.get(
        "dependency-groups",
    )

    if isinstance(
        dependency_groups,
        dict,
    ):
        for group in dependency_groups.values():
            _add_python_dependencies(
                dependencies,
                group,
            )

    return dependencies


def _minimum_version(
    specification: str,
) -> tuple[str, bool] | None:
    """Return a displayable dependency version."""

    for pattern, minimum in (
        (
            MINIMUM_VERSION_PATTERN,
            True,
        ),
        (
            EXACT_VERSION_PATTERN,
            False,
        ),
        (
            COMPATIBLE_VERSION_PATTERN,
            True,
        ),
    ):
        match = pattern.search(
            specification,
        )

        if match is not None:
            return (
                match.group(1),
                minimum,
            )

    return None


def _dependency_badge_version(
    specification: str,
) -> str | None:
    """Convert a dependency specification into badge format."""

    result = _minimum_version(
        specification,
    )

    if result is None:
        return None

    version, minimum = result

    if minimum:
        return f"{version}+"

    return version


def _strip_version_prefix(
    value: str,
) -> str:
    """Remove a leading version prefix."""

    return value.removeprefix(
        "v",
    ).strip()


def _get_uv_version(
    pyproject: Config,
) -> str | None:
    """Return the configured uv requirement."""

    tool = pyproject.get(
        "tool",
    )

    if not isinstance(
        tool,
        dict,
    ):
        return None

    uv = tool.get(
        "uv",
    )

    if not isinstance(
        uv,
        dict,
    ):
        return None

    value = uv.get(
        "required-version",
    )

    if not isinstance(
        value,
        str,
    ):
        value = uv.get(
            "required_version",
        )

    if not isinstance(
        value,
        str,
    ):
        return None

    return _dependency_badge_version(
        value.strip(),
    )


def _get_engine_version(
    package_json: Config,
    engine_name: str,
) -> str | None:
    """Return a package.json engine version."""

    engines = package_json.get(
        "engines",
    )

    if not isinstance(
        engines,
        dict,
    ):
        return None

    specification = engines.get(
        engine_name,
    )

    if not isinstance(
        specification,
        str,
    ):
        return None

    return _dependency_badge_version(
        specification.strip(),
    )


def _get_npm_version(
    package_json: Config,
) -> str | None:
    """Return the canonical npm version."""

    package_manager = package_json.get(
        "packageManager",
    )

    if isinstance(
        package_manager,
        str,
    ):
        match = NPM_PACKAGE_MANAGER_PATTERN.fullmatch(
            package_manager.strip(),
        )

        if match is not None:
            return match.group(
                1,
            ).strip()

    return _get_engine_version(
        package_json,
        "npm",
    )


def _get_node_version(
    package_json: Config,
) -> str:
    """Return the canonical Node.js version."""

    if NODE_VERSION_PATH.is_file():
        return _strip_version_prefix(
            _read_text_value(
                NODE_VERSION_PATH,
            )
        )

    version = _get_engine_version(
        package_json,
        "node",
    )

    if version is not None:
        return version

    msg = "Node.js version was not found in .nvmrc or package.json."

    raise ValueError(msg)


def _get_javascript_dependency_version(
    package_json: Config,
    package_name: str,
) -> str | None:
    """Return a JavaScript dependency version."""

    for section_name in (
        "dependencies",
        "devDependencies",
        "optionalDependencies",
    ):
        section = package_json.get(
            section_name,
        )

        if not isinstance(
            section,
            dict,
        ):
            continue

        value = section.get(
            package_name,
        )

        if not isinstance(
            value,
            str,
        ):
            continue

        return value.strip().lstrip(
            "^~",
        )

    return None
