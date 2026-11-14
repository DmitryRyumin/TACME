"""README generator implementation for the TACME project.

Original authors: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin.
License: MIT License.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

type Config = dict[str, Any]
type DependencyMap = dict[str, str]


ROOT = Path(__file__).resolve().parents[2]

PYPROJECT_PATH = ROOT / "pyproject.toml"
PACKAGE_JSON_PATH = ROOT / "package.json"
APP_CONFIG_PATH = ROOT / "config.toml"

PYTHON_VERSION_PATH = ROOT / ".python-version"
NODE_VERSION_PATH = ROOT / ".nvmrc"

DOCKERFILE_PATH = ROOT / "Dockerfile"
README_PATH = ROOT / "README.md"

README_BLOCK_START = "<!-- BEGIN PROJECT BADGES -->"
README_BLOCK_END = "<!-- END PROJECT BADGES -->"

AUTHORS_BLOCK_START = "<!-- BEGIN AUTHORS -->"
AUTHORS_BLOCK_END = "<!-- END AUTHORS -->"

PRETTIER_IGNORE_START = "<!-- prettier-ignore-start -->"
PRETTIER_IGNORE_END = "<!-- prettier-ignore-end -->"

GITHUB_OWNER = "DmitryRyumin"
HUGGING_FACE_OWNER = GITHUB_OWNER

AUTHORS_IMAGE_DIRECTORY = Path(
    "static",
    "images",
    "authors",
    "generated",
)

AUTHOR_LINK_ORDER = (
    "LINK_GITHUB",
    "LINK_WEBSITE",
    "LINK_WEB_OF_SCIENCE",
    "LINK_SCOPUS",
    "LINK_GOOGLE_SCHOLAR",
    "LINK_ORCID",
    "LINK_HUGGING_FACE",
    "LINK_TELEGRAM",
    "LINK_EMAIL",
)

PYTHON_DEPENDENCY_PATTERN = re.compile(
    r"^([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?\s*(.*)$",
)

MINIMUM_VERSION_PATTERN = re.compile(
    r">=\s*([0-9]+(?:\.[0-9A-Za-z]+)*)",
)

EXACT_VERSION_PATTERN = re.compile(
    r"==\s*([0-9]+(?:\.[0-9A-Za-z]+)*)",
)

COMPATIBLE_VERSION_PATTERN = re.compile(
    r"~=\s*([0-9]+(?:\.[0-9A-Za-z]+)*)",
)

NPM_PACKAGE_MANAGER_PATTERN = re.compile(
    r"npm@(.+)",
)
