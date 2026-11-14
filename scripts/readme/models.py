"""README generator implementation for the TACME project.

Original authors: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin.
License: MIT License.
"""

from __future__ import annotations

from dataclasses import dataclass

from scripts.readme.constants import (
    GITHUB_OWNER,
    HUGGING_FACE_OWNER,
    Config,
)
from scripts.readme.sources import (
    _get_project_name,
)


@dataclass(
    frozen=True,
    slots=True,
)
class ProjectMetadata:
    """Canonical TACME repository and application metadata."""

    package_name: str
    display_name: str
    github_repository: str
    hugging_face_space: str
    hugging_face_app_url: str

    @classmethod
    def from_pyproject(
        cls,
        pyproject: Config,
    ) -> ProjectMetadata:
        """Create project metadata from pyproject.toml."""

        package_name = _get_project_name(
            pyproject,
        )

        display_name = package_name.upper()

        github_repository = f"{GITHUB_OWNER}/{display_name}"

        hugging_face_space = f"{HUGGING_FACE_OWNER}/{display_name}"

        app_slug = (
            (f"{HUGGING_FACE_OWNER}-{package_name}")
            .lower()
            .replace(
                "_",
                "-",
            )
        )

        return cls(
            package_name=package_name,
            display_name=display_name,
            github_repository=github_repository,
            hugging_face_space=hugging_face_space,
            hugging_face_app_url=(f"https://{app_slug}.hf.space"),
        )


@dataclass(
    frozen=True,
    slots=True,
)
class ReadmeAuthorLink:
    """One author link rendered in the README."""

    label: str
    url: str


@dataclass(
    frozen=True,
    slots=True,
)
class ReadmeAuthor:
    """Author metadata rendered in the TACME README."""

    name: str
    degree: str
    initials: str
    photo_stem: str
    links: tuple[ReadmeAuthorLink, ...]
