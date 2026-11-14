"""README generator implementation for the TACME project.

Original authors: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin.
License: MIT License.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib

from scripts.readme.constants import (
    PYPROJECT_PATH,
)
from scripts.readme.models import (
    ProjectMetadata,
)
from scripts.readme.rendering import (
    _update_readme,
)
from scripts.readme.sources import (
    _load_toml,
)


def _parse_arguments(
    project_name: str,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(f"Generate the technical metadata and Authors section in the {project_name} README."),
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help=("Check generated README content without modifying README.md."),
    )

    return parser.parse_args()


def main() -> int:
    """Run the README generator."""

    try:
        pyproject = _load_toml(
            PYPROJECT_PATH,
        )

        metadata = ProjectMetadata.from_pyproject(
            pyproject,
        )

        arguments = _parse_arguments(
            metadata.display_name,
        )

        success = _update_readme(
            check=arguments.check,
        )

    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
    ) as error:
        print(
            f"README generation failed: {error}",
            file=sys.stderr,
        )

        return 1

    return 0 if success else 1
