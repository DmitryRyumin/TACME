"""README generator implementation for the TACME project.

Original authors: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin.
License: MIT License.
"""

from __future__ import annotations

import html

from scripts.readme.badge_elements import (
    Badge,
    _image,
    _javascript_dependency_badge,
    _link,
    _linked_badge,
    _python_dependency_badge,
)
from scripts.readme.constants import (
    DOCKERFILE_PATH,
    PACKAGE_JSON_PATH,
    PRETTIER_IGNORE_END,
    PRETTIER_IGNORE_START,
    PYPROJECT_PATH,
    PYTHON_VERSION_PATH,
    Config,
    DependencyMap,
)
from scripts.readme.dependencies import (
    _collect_python_dependencies,
    _get_node_version,
    _get_npm_version,
    _get_uv_version,
    _strip_version_prefix,
)
from scripts.readme.models import (
    ProjectMetadata,
)
from scripts.readme.sources import (
    _get_project_version,
    _load_json,
    _load_toml,
    _read_text_value,
)


def _existing_badges(
    badges: tuple[str | None, ...],
) -> list[str]:
    """Remove absent optional badges."""

    return [badge for badge in badges if badge is not None]


def _build_project_badges(
    pyproject: Config,
    metadata: ProjectMetadata,
) -> list[str]:
    """Build project metadata badges."""

    python_version = _strip_version_prefix(
        _read_text_value(
            PYTHON_VERSION_PATH,
        )
    )

    return [
        Badge(
            label="version",
            message=_get_project_version(
                pyproject,
            ),
            color="2F81F7",
            alt="Version",
        ).render(),
        _link(
            "LICENSE",
            _image(
                (f"https://img.shields.io/github/license/{metadata.github_repository}?style=flat-square"),
                "License",
            ),
        ),
        Badge(
            label="Python",
            message=python_version,
            color="3776AB",
            alt="Python",
            logo="python",
            logo_color="white",
        ).render(),
    ]


def _build_runtime_badges(
    pyproject: Config,
    package_json: Config,
    dependencies: DependencyMap,
) -> list[str]:
    """Build TACME runtime technology badges."""

    badges = _existing_badges(
        (
            _python_dependency_badge(
                dependencies,
                "gradio",
                label="Gradio",
                color="F97316",
            ),
            _python_dependency_badge(
                dependencies,
                "torch",
                label="PyTorch",
                color="EE4C2C",
                logo="pytorch",
                logo_color="white",
            ),
            _python_dependency_badge(
                dependencies,
                "torchcodec",
                label="TorchCodec",
                color="EE4C2C",
            ),
            _python_dependency_badge(
                dependencies,
                "transformers",
                label="Transformers",
                color="FFD21E",
            ),
            _python_dependency_badge(
                dependencies,
                "accelerate",
                label="Accelerate",
                color="FFD21E",
            ),
            _python_dependency_badge(
                dependencies,
                "safetensors",
                label="Safetensors",
                color="7C3AED",
            ),
            _python_dependency_badge(
                dependencies,
                "packaging",
                label="packaging",
                color="6B7280",
            ),
            _python_dependency_badge(
                dependencies,
                "psutil",
                label="psutil",
                color="4B5563",
            ),
        )
    )

    if DOCKERFILE_PATH.is_file():
        badges.append(
            Badge(
                label="Docker",
                message="enabled",
                color="2496ED",
                alt="Docker",
                logo="docker",
                logo_color="white",
            ).render()
        )

    uv_version = _get_uv_version(
        pyproject,
    )

    if uv_version is not None:
        badges.append(
            Badge(
                label="uv",
                message=uv_version,
                color="261230",
                alt="uv",
            ).render()
        )

    badges.append(
        Badge(
            label="Node.js",
            message=_get_node_version(
                package_json,
            ),
            color="5FA04E",
            alt="Node.js",
            logo="nodedotjs",
            logo_color="white",
        ).render()
    )

    npm_version = _get_npm_version(
        package_json,
    )

    if npm_version is not None:
        badges.append(
            Badge(
                label="npm",
                message=npm_version,
                color="CB3837",
                alt="npm",
                logo="npm",
                logo_color="white",
            ).render()
        )

    return badges


def _build_build_tooling_badges(
    package_json: Config,
) -> list[str]:
    """Build frontend and asset build-tool badges."""

    return _existing_badges(
        (
            _javascript_dependency_badge(
                package_json,
                "sharp",
                label="Sharp",
                color="99CC00",
            ),
            _javascript_dependency_badge(
                package_json,
                "lightningcss-cli",
                label="Lightning CSS",
                color="FACC15",
            ),
            _javascript_dependency_badge(
                package_json,
                "@taplo/cli",
                label="Taplo",
                color="6B7280",
            ),
        )
    )


def _build_code_quality_badges(
    package_json: Config,
    dependencies: DependencyMap,
) -> list[str]:
    """Build code-quality tool badges."""

    return _existing_badges(
        (
            _python_dependency_badge(
                dependencies,
                "ruff",
                label="Ruff",
                color="D7FF64",
            ),
            _python_dependency_badge(
                dependencies,
                "mypy",
                label="mypy",
                color="2A6DB2",
            ),
            _python_dependency_badge(
                dependencies,
                "deptry",
                label="deptry",
                color="6B7280",
            ),
            _python_dependency_badge(
                dependencies,
                "pre-commit",
                label="pre-commit",
                color="FAB040",
            ),
            _javascript_dependency_badge(
                package_json,
                "stylelint",
                label="Stylelint",
                color="263238",
            ),
            _javascript_dependency_badge(
                package_json,
                "prettier",
                label="Prettier",
                color="F7B93E",
                logo="prettier",
                logo_color="black",
            ),
        )
    )


def _github_badge_url(
    metric: str,
    metadata: ProjectMetadata,
) -> str:
    """Build a live GitHub Shields.io badge URL."""

    return f"https://img.shields.io/github/{metric}/{metadata.github_repository}?style=flat-square"


def _github_url(
    metadata: ProjectMetadata,
    suffix: str = "",
) -> str:
    """Build a GitHub repository URL."""

    return f"https://github.com/{metadata.github_repository}{suffix}"


def _build_repository_badges(
    metadata: ProjectMetadata,
) -> list[str]:
    """Build repository activity badges."""

    return [
        _image(
            _github_badge_url(
                "repo-size",
                metadata,
            ),
            "Repository size",
        ),
        _image(
            _github_badge_url(
                "last-commit",
                metadata,
            ),
            "Last commit",
        ),
        _link(
            _github_url(
                metadata,
                "/graphs/contributors",
            ),
            _image(
                _github_badge_url(
                    "contributors",
                    metadata,
                ),
                "Contributors",
            ),
        ),
    ]


def _build_community_badges(
    metadata: ProjectMetadata,
) -> list[str]:
    """Build GitHub community badges."""

    return [
        _link(
            _github_url(
                metadata,
                "/stargazers",
            ),
            _image(
                _github_badge_url(
                    "stars",
                    metadata,
                ),
                "Stars",
            ),
        ),
        _link(
            _github_url(
                metadata,
                "/forks",
            ),
            _image(
                _github_badge_url(
                    "forks",
                    metadata,
                ),
                "Forks",
            ),
        ),
        _link(
            _github_url(
                metadata,
                "/issues",
            ),
            _image(
                _github_badge_url(
                    "issues",
                    metadata,
                ),
                "Issues",
            ),
        ),
    ]


def _build_application_badges(
    metadata: ProjectMetadata,
) -> list[str]:
    """Build TACME application badges."""

    return [
        _linked_badge(
            metadata.hugging_face_app_url,
            Badge(
                label="🤗 Space",
                message=metadata.display_name,
                color="FFD21E",
                alt=(f"{metadata.display_name} on Hugging Face"),
            ),
        ),
    ]


def _table_row(
    title: str,
    badges: list[str],
) -> list[str]:
    """Build one README metadata table row."""

    title_html = html.escape(
        title,
    ).replace(
        " ",
        "&nbsp;",
    )

    return [
        "  <tr>",
        (f'    <td width="150"><strong>{title_html}</strong></td>'),
        f"    <td>{' '.join(badges)}</td>",
        "  </tr>",
    ]


def _build_badge_block() -> str:
    """Build the generated README metadata table."""

    pyproject = _load_toml(
        PYPROJECT_PATH,
    )

    package_json = _load_json(
        PACKAGE_JSON_PATH,
    )

    metadata = ProjectMetadata.from_pyproject(
        pyproject,
    )

    dependencies = _collect_python_dependencies(
        pyproject,
    )

    rows = [
        PRETTIER_IGNORE_START,
        '<table align="center">',
    ]

    sections = (
        (
            "Project",
            _build_project_badges(
                pyproject,
                metadata,
            ),
        ),
        (
            "Runtime",
            _build_runtime_badges(
                pyproject,
                package_json,
                dependencies,
            ),
        ),
        (
            "Build Tooling",
            _build_build_tooling_badges(
                package_json,
            ),
        ),
        (
            "Code Quality",
            _build_code_quality_badges(
                package_json,
                dependencies,
            ),
        ),
        (
            "Repository",
            _build_repository_badges(
                metadata,
            ),
        ),
        (
            "Community",
            _build_community_badges(
                metadata,
            ),
        ),
        (
            "Application",
            _build_application_badges(
                metadata,
            ),
        ),
    )

    for title, badges in sections:
        if not badges:
            continue

        rows.extend(
            _table_row(
                title,
                badges,
            )
        )

    rows.extend(
        (
            "</table>",
            PRETTIER_IGNORE_END,
        )
    )

    return "\n".join(
        rows,
    )
