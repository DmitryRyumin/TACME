"""README generator implementation for the TACME project.

Original authors: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin.
License: MIT License.
"""

from __future__ import annotations

import json
from pathlib import Path
import tomllib

from scripts.readme.constants import (
    Config,
)


def _load_toml(
    path: Path,
) -> Config:
    """Load a TOML document."""

    with path.open("rb") as file:
        data = tomllib.load(file)

    if not isinstance(
        data,
        dict,
    ):
        msg = f"Expected a TOML table in {path}."
        raise TypeError(msg)

    return data


def _load_json(
    path: Path,
) -> Config:
    """Load a JSON object."""

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(
        data,
        dict,
    ):
        msg = f"Expected a JSON object in {path}."
        raise TypeError(msg)

    return data


def _read_text_value(
    path: Path,
) -> str:
    """Read a required one-line text value."""

    value = path.read_text(
        encoding="utf-8",
    ).strip()

    if not value:
        msg = f"Expected a non-empty value in {path}."
        raise ValueError(msg)

    return value


def _get_project_table(
    pyproject: Config,
) -> Config:
    """Return the required [project] table."""

    project = pyproject.get(
        "project",
    )

    if not isinstance(
        project,
        dict,
    ):
        msg = "Missing [project] table in pyproject.toml."
        raise ValueError(msg)

    return project


def _get_required_string(
    mapping: Config,
    key: str,
    *,
    source: str,
) -> str:
    """Return a required non-empty string."""

    value = mapping.get(
        key,
    )

    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        msg = f"Missing {key!r} in {source}."
        raise ValueError(msg)

    return value.strip()


def _get_project_name(
    pyproject: Config,
) -> str:
    """Return the canonical project package name."""

    return _get_required_string(
        _get_project_table(
            pyproject,
        ),
        "name",
        source="pyproject.toml [project]",
    )


def _get_project_version(
    pyproject: Config,
) -> str:
    """Return the project version."""

    return _get_required_string(
        _get_project_table(
            pyproject,
        ),
        "version",
        source="pyproject.toml [project]",
    )
