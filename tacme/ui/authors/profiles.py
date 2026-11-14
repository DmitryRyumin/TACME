"""
File: profiles.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Localized TACME author profiles and photo resolution.

License: MIT License
"""

from pathlib import Path
from urllib.parse import quote

from tacme.config import PROJECT_ROOT, get_config_str, get_config_str_list, get_config_str_mapping
from tacme.localization import get_localized_text
from tacme.ui.authors.contracts import (
    AUTHOR_LINK_LABEL_KEYS,
    AUTHOR_LINK_ORDER,
    AuthorLink,
    AuthorPhotoSources,
    AuthorProfile,
)


def _create_author_link_label(
    config_key: str,
    url: str,
    language_index: int,
) -> str:
    """Create the localized visible label for one author profile link."""

    localization_key = AUTHOR_LINK_LABEL_KEYS.get(config_key)

    label = (
        get_localized_text(
            localization_key,
            language_index,
        )
        if localization_key is not None
        else config_key
    )

    if config_key != "LINK_ORCID":
        return label

    orcid_identifier = (
        url.rstrip("/")
        .rsplit(
            "/",
            maxsplit=1,
        )[-1]
        .strip()
    )

    if not orcid_identifier:
        return label

    return f"{label} · {orcid_identifier}"


def _create_gradio_file_url(
    path: Path,
) -> str:
    """Create a Gradio file URL for a local static asset."""

    encoded_path = quote(
        str(path),
        safe="/:",
    )

    return f"/gradio_api/file={encoded_path}"


def _resolve_author_photo(
    values: dict[str, str],
) -> AuthorPhotoSources | None:
    """Resolve optimized author photo assets."""

    photo_stem = values.get(
        "PHOTO",
        "",
    ).strip()

    if not photo_stem:
        return None

    if Path(photo_stem).name != photo_stem:
        return None

    images_root = (
        PROJECT_ROOT
        / get_config_str(
            "StaticPaths_IMAGES",
            "static/images",
        )
    ).resolve()

    generated_directory = images_root / "authors" / "generated"

    jpeg_path = generated_directory / f"{photo_stem}.jpg"

    if not jpeg_path.is_file():
        return None

    avif_path = generated_directory / f"{photo_stem}.avif"

    webp_path = generated_directory / f"{photo_stem}.webp"

    return AuthorPhotoSources(
        avif_url=(_create_gradio_file_url(avif_path) if avif_path.is_file() else None),
        webp_url=(_create_gradio_file_url(webp_path) if webp_path.is_file() else None),
        jpeg_url=_create_gradio_file_url(
            jpeg_path,
        ),
    )


def _get_author_profile(
    author_key: str,
    language_index: int,
) -> AuthorProfile:
    """Load one localized author profile from config.toml."""

    section_name = f"Author_{author_key}"

    values = get_config_str_mapping(
        section_name,
    )

    name = get_localized_text(
        f"Authors_NAME_{author_key}",
        language_index,
    ).strip()

    degree = get_localized_text(
        f"Authors_DEGREE_{author_key}",
        language_index,
    ).strip()

    initials = values.get(
        "INITIALS",
        "",
    ).strip()

    if not name:
        msg = f"Author '{author_key}' must define Authors_NAME_{author_key}."
        raise ValueError(msg)

    if not initials:
        msg = f"Author configuration section '{section_name}' must define INITIALS."
        raise ValueError(msg)

    links: list[AuthorLink] = []

    for config_key in AUTHOR_LINK_ORDER:
        url = values.get(
            config_key,
            "",
        ).strip()

        if not url:
            continue

        links.append(
            AuthorLink(
                label=_create_author_link_label(
                    config_key,
                    url,
                    language_index,
                ),
                url=url,
            )
        )

    return AuthorProfile(
        name=name,
        degree=degree,
        initials=initials,
        photo=_resolve_author_photo(
            values,
        ),
        links=tuple(links),
    )


def get_author_profiles(
    language_index: int,
) -> tuple[AuthorProfile, ...]:
    """Return localized TACME authors in configured display order."""

    author_keys = get_config_str_list(
        "Authors_ORDER",
        [],
    )

    if not author_keys:
        msg = "Authors_ORDER must contain at least one configured author."
        raise ValueError(msg)

    return tuple(
        _get_author_profile(
            author_key,
            language_index,
        )
        for author_key in author_keys
    )
