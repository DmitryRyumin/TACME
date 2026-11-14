"""
File: contracts.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Author metadata and component contracts.

License: MIT License
"""

from dataclasses import dataclass
from typing import Final

import gradio as gr

AUTHOR_LINK_ORDER: Final[tuple[str, ...]] = (
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


AUTHOR_LINK_LABEL_KEYS: Final[dict[str, str]] = {
    "LINK_GITHUB": "Authors_LINK_GITHUB",
    "LINK_WEBSITE": "Authors_LINK_WEBSITE",
    "LINK_WEB_OF_SCIENCE": "Authors_LINK_WEB_OF_SCIENCE",
    "LINK_SCOPUS": "Authors_LINK_SCOPUS",
    "LINK_GOOGLE_SCHOLAR": "Authors_LINK_GOOGLE_SCHOLAR",
    "LINK_ORCID": "Authors_LINK_ORCID",
    "LINK_HUGGING_FACE": "Authors_LINK_HUGGING_FACE",
    "LINK_TELEGRAM": "Authors_LINK_TELEGRAM",
    "LINK_EMAIL": "Authors_LINK_EMAIL",
}


@dataclass(
    frozen=True,
    slots=True,
)
class AuthorLink:
    """One external author profile link."""

    label: str
    url: str


@dataclass(
    frozen=True,
    slots=True,
)
class AuthorPhotoSources:
    """Optimized browser image sources for one author."""

    avif_url: str | None
    webp_url: str | None
    jpeg_url: str


@dataclass(
    frozen=True,
    slots=True,
)
class AuthorProfile:
    """One localized TACME author profile."""

    name: str
    degree: str
    initials: str
    photo: AuthorPhotoSources | None
    links: tuple[AuthorLink, ...]


@dataclass(
    frozen=True,
    slots=True,
)
class AuthorsTabComponents:
    """Components created inside the TACME Authors tab."""

    content: gr.HTML
