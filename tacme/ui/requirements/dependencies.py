"""
File: dependencies.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: TACME Requirements tab implementation.

License: MIT License
"""

from importlib import metadata
import json
import tomllib
from typing import Final, cast
from urllib.parse import quote
from urllib.request import Request, urlopen

from packaging.requirements import InvalidRequirement, Requirement

from tacme.config import PROJECT_ROOT, get_config_str
from tacme.logger import get_logger
from tacme.ui.requirements.models import RuntimeDependency

LOGGER = get_logger("tacme.ui.requirements")


PYPROJECT_PATH: Final = PROJECT_ROOT / "pyproject.toml"


PYPI_JSON_URL_TEMPLATE: Final = "https://pypi.org/pypi/{package_name}/json"


PYPI_TIMEOUT_SECONDS: Final = 4.0


def _read_project_metadata() -> tuple[
    str,
    list[str],
]:
    """Read Python and runtime requirements from pyproject.toml."""

    with PYPROJECT_PATH.open("rb") as file:
        data = tomllib.load(file)

    project_value = data.get(
        "project",
    )

    if not isinstance(
        project_value,
        dict,
    ):
        msg = "pyproject.toml must contain a [project] table."
        raise TypeError(msg)

    project = cast(
        dict[str, object],
        project_value,
    )

    requires_python_value = project.get(
        "requires-python",
        "",
    )

    requires_python = (
        requires_python_value
        if isinstance(
            requires_python_value,
            str,
        )
        else ""
    )

    dependencies_value = project.get(
        "dependencies",
        [],
    )

    if not isinstance(
        dependencies_value,
        list,
    ) or not all(
        isinstance(
            dependency,
            str,
        )
        for dependency in dependencies_value
    ):
        msg = "pyproject.toml project.dependencies must be a list of strings."
        raise TypeError(msg)

    return (
        requires_python,
        cast(
            list[str],
            dependencies_value,
        ),
    )


def _get_installed_version(
    package_name: str,
) -> str | None:
    """Return the installed distribution version."""

    try:
        return metadata.version(
            package_name,
        )

    except metadata.PackageNotFoundError:
        return None


def get_runtime_dependencies() -> tuple[
    str,
    tuple[RuntimeDependency, ...],
]:
    """Resolve direct TACME dependencies from the active environment."""

    requires_python, raw_dependencies = _read_project_metadata()

    dependencies: list[RuntimeDependency] = []

    for raw_requirement in raw_dependencies:
        try:
            requirement = Requirement(
                raw_requirement,
            )

        except InvalidRequirement as error:
            msg = f"Invalid runtime requirement in pyproject.toml: {raw_requirement!r}."

            raise ValueError(msg) from error

        dependencies.append(
            RuntimeDependency(
                requirement=requirement,
                raw_requirement=raw_requirement,
                installed_version=_get_installed_version(
                    requirement.name,
                ),
            )
        )

    return (
        requires_python,
        tuple(dependencies),
    )


def _fetch_latest_pypi_version(
    package_name: str,
) -> str | None:
    """Fetch the latest published package version from PyPI."""

    package_url_name = quote(
        package_name,
        safe="",
    )

    app_version = get_config_str(
        "App_VERSION",
        "1.0.0",
    )

    request = Request(
        PYPI_JSON_URL_TEMPLATE.format(
            package_name=package_url_name,
        ),
        headers={
            "Accept": "application/json",
            "User-Agent": f"TACME/{app_version}",
        },
    )

    try:
        with urlopen(
            request,
            timeout=PYPI_TIMEOUT_SECONDS,
        ) as response:
            payload_value: object = json.loads(
                response.read().decode(
                    "utf-8",
                )
            )

    except (
        OSError,
        TimeoutError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ) as error:
        LOGGER.warning(
            "Could not retrieve the latest PyPI version for '%s': %s",
            package_name,
            error,
        )

        return None

    if not isinstance(
        payload_value,
        dict,
    ):
        return None

    payload = cast(
        dict[str, object],
        payload_value,
    )

    info_value = payload.get(
        "info",
    )

    if not isinstance(
        info_value,
        dict,
    ):
        return None

    info = cast(
        dict[str, object],
        info_value,
    )

    version_value = info.get(
        "version",
    )

    if (
        not isinstance(
            version_value,
            str,
        )
        or not version_value.strip()
    ):
        return None

    return version_value.strip()
