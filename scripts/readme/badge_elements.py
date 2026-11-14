"""README generator implementation for the TACME project.

Original authors: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin.
License: MIT License.
"""

from __future__ import annotations

from dataclasses import dataclass
import html
from urllib.parse import quote

from scripts.readme.constants import (
    Config,
    DependencyMap,
)
from scripts.readme.dependencies import (
    _dependency_badge_version,
    _get_javascript_dependency_version,
    _normalize_package_name,
)


@dataclass(
    frozen=True,
    slots=True,
)
class Badge:
    """Description of one static Shields.io badge."""

    label: str
    message: str
    color: str
    alt: str
    logo: str | None = None
    logo_color: str | None = None

    def render(
        self,
    ) -> str:
        """Render the badge as an HTML image."""

        return _image(
            _badge_url(
                self.label,
                self.message,
                self.color,
                logo=self.logo,
                logo_color=self.logo_color,
            ),
            self.alt,
        )


def _encode_badge_part(
    value: str,
) -> str:
    """Encode text for a Shields.io static badge URL."""

    return quote(
        value,
        safe="",
    ).replace(
        "-",
        "--",
    )


def _badge_url(
    label: str,
    message: str,
    color: str,
    *,
    logo: str | None = None,
    logo_color: str | None = None,
) -> str:
    """Build a Shields.io static badge URL."""

    url = (
        "https://img.shields.io/badge/"
        f"{_encode_badge_part(label)}-"
        f"{_encode_badge_part(message)}-"
        f"{color}"
        "?style=flat-square"
    )

    if logo is not None:
        url += f"&logo={quote(logo, safe='')}"

    if logo_color is not None:
        url += f"&logoColor={quote(logo_color, safe='')}"

    return url


def _image(
    source: str,
    alt: str,
) -> str:
    """Build an HTML image element."""

    return f'<img src="{html.escape(source, quote=True)}" alt="{html.escape(alt, quote=True)}">'


def _link(
    href: str,
    content: str,
) -> str:
    """Build an HTML anchor element."""

    return f'<a href="{html.escape(href, quote=True)}">{content}</a>'


def _linked_badge(
    href: str,
    badge: Badge,
) -> str:
    """Render a linked badge."""

    return _link(
        href,
        badge.render(),
    )


def _python_dependency_badge(
    dependencies: DependencyMap,
    package_name: str,
    *,
    label: str,
    color: str,
    logo: str | None = None,
    logo_color: str | None = None,
) -> str | None:
    """Build a Python dependency badge."""

    specification = dependencies.get(
        _normalize_package_name(
            package_name,
        )
    )

    if specification is None:
        return None

    version = _dependency_badge_version(
        specification,
    )

    if version is None:
        return None

    return Badge(
        label=label,
        message=version,
        color=color,
        alt=label,
        logo=logo,
        logo_color=logo_color,
    ).render()


def _javascript_dependency_badge(
    package_json: Config,
    package_name: str,
    *,
    label: str,
    color: str,
    logo: str | None = None,
    logo_color: str | None = None,
) -> str | None:
    """Build a JavaScript dependency badge."""

    version = _get_javascript_dependency_version(
        package_json,
        package_name,
    )

    if version is None:
        return None

    return Badge(
        label=label,
        message=version,
        color=color,
        alt=label,
        logo=logo,
        logo_color=logo_color,
    ).render()
