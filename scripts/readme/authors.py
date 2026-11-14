"""README generator implementation for the TACME project.

Original authors: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin.
License: MIT License.
"""

from __future__ import annotations

import html

from scripts.readme.constants import (
    APP_CONFIG_PATH,
    AUTHOR_LINK_ORDER,
    AUTHORS_IMAGE_DIRECTORY,
    PRETTIER_IGNORE_END,
    PRETTIER_IGNORE_START,
    ROOT,
    Config,
)
from scripts.readme.models import (
    ReadmeAuthor,
    ReadmeAuthorLink,
)
from scripts.readme.sources import (
    _load_toml,
)


def _get_config_table(
    config: Config,
    table_name: str,
) -> Config:
    """Return one required TOML table."""

    table = config.get(
        table_name,
    )

    if not isinstance(
        table,
        dict,
    ):
        msg = f"Missing or invalid [{table_name}] table in config.toml."
        raise ValueError(msg)

    return table


def _get_localized_config_value(
    mapping: Config,
    key: str,
    language_index: int = 0,
) -> str:
    """Return one localized string from a configuration value."""

    value = mapping.get(
        key,
    )

    if not isinstance(
        value,
        list,
    ):
        return ""

    if language_index < 0 or language_index >= len(
        value,
    ):
        return ""

    localized_value = value[language_index]

    if not isinstance(
        localized_value,
        str,
    ):
        return ""

    return localized_value.strip()


def _get_author_link_label(
    authors_table: Config,
    config_key: str,
    url: str,
) -> str:
    """Return the English README label for one author link."""

    label = _get_localized_config_value(
        authors_table,
        config_key,
    )

    if not label:
        label = (
            config_key.removeprefix(
                "LINK_",
            )
            .replace(
                "_",
                " ",
            )
            .title()
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


def _get_readme_authors(
    config: Config,
) -> tuple[ReadmeAuthor, ...]:
    """Load README author metadata from config.toml."""

    authors_table = _get_config_table(
        config,
        "Authors",
    )

    order_value = authors_table.get(
        "ORDER",
    )

    if not isinstance(
        order_value,
        list,
    ) or not all(
        isinstance(
            author_key,
            str,
        )
        for author_key in order_value
    ):
        msg = "Authors.ORDER must be a list of strings."
        raise ValueError(msg)

    authors: list[ReadmeAuthor] = []

    for author_key_value in order_value:
        author_key = str(
            author_key_value,
        )

        author_table = _get_config_table(
            config,
            f"Author_{author_key}",
        )

        name = _get_localized_config_value(
            authors_table,
            f"NAME_{author_key}",
        )

        degree = _get_localized_config_value(
            authors_table,
            f"DEGREE_{author_key}",
        )

        initials_value = author_table.get(
            "INITIALS",
            "",
        )

        photo_value = author_table.get(
            "PHOTO",
            "",
        )

        initials = (
            initials_value.strip()
            if isinstance(
                initials_value,
                str,
            )
            else ""
        )

        photo_stem = (
            photo_value.strip()
            if isinstance(
                photo_value,
                str,
            )
            else ""
        )

        if not name:
            msg = f"Author '{author_key}' must define an English name."
            raise ValueError(msg)

        if not initials:
            msg = f"Author '{author_key}' must define INITIALS."
            raise ValueError(msg)

        links: list[ReadmeAuthorLink] = []

        for config_key in AUTHOR_LINK_ORDER:
            url_value = author_table.get(
                config_key,
                "",
            )

            if not isinstance(
                url_value,
                str,
            ):
                continue

            url = url_value.strip()

            if not url:
                continue

            links.append(
                ReadmeAuthorLink(
                    label=_get_author_link_label(
                        authors_table,
                        config_key,
                        url,
                    ),
                    url=url,
                )
            )

        authors.append(
            ReadmeAuthor(
                name=name,
                degree=degree,
                initials=initials,
                photo_stem=photo_stem,
                links=tuple(
                    links,
                ),
            )
        )

    return tuple(
        authors,
    )


def _create_readme_author_photo(
    author: ReadmeAuthor,
) -> str:
    """Create an author photo or initials fallback."""

    if author.photo_stem:
        relative_path = AUTHORS_IMAGE_DIRECTORY / f"{author.photo_stem}.jpg"

        absolute_path = ROOT / relative_path

        if absolute_path.is_file():
            source = relative_path.as_posix()

            return (
                "<img "
                f'src="{html.escape(source, quote=True)}" '
                f'alt="{html.escape(author.name, quote=True)}" '
                'width="96" '
                'height="96">'
            )

    return f"<strong>{html.escape(author.initials)}</strong>"


def _create_readme_author_links(
    author: ReadmeAuthor,
) -> str:
    """Create compact links for one README author."""

    links = [(f'<a href="{html.escape(link.url, quote=True)}">{html.escape(link.label)}</a>') for link in author.links]

    return "&nbsp;·&nbsp;".join(
        links,
    )


def _create_readme_author_cell(
    author: ReadmeAuthor,
) -> str:
    """Create one centered README author card cell."""

    photo_html = _create_readme_author_photo(
        author,
    )

    links_html = _create_readme_author_links(
        author,
    )

    display_name = " ".join(
        part
        for part in (
            author.degree.strip(),
            author.name.strip(),
        )
        if part
    )

    links_block = (f"<br><br><sub>{links_html}</sub>") if links_html else ""

    return (
        '<td align="center" '
        'valign="top" '
        'width="33%">'
        f"{photo_html}"
        "<br>"
        "<strong>"
        f"{html.escape(display_name)}"
        "</strong>"
        f"{links_block}"
        "</td>"
    )


def _build_authors_block() -> str:
    """Build the generated TACME README Authors section."""

    config = _load_toml(
        APP_CONFIG_PATH,
    )

    authors = _get_readme_authors(
        config,
    )

    if not authors:
        return ""

    rows: list[str] = [
        PRETTIER_IGNORE_START,
        '<table align="center">',
    ]

    for row_start in range(
        0,
        len(authors),
        3,
    ):
        row_authors = authors[row_start : row_start + 3]

        rows.append(
            "  <tr>",
        )

        for author in row_authors:
            rows.append(
                "    "
                + _create_readme_author_cell(
                    author,
                )
            )

        rows.append(
            "  </tr>",
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
