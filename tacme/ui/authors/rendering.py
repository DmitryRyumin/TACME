"""
File: rendering.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: HTML rendering for TACME Authors cards.

License: MIT License
"""

from html import escape

from tacme.ui.authors.contracts import AuthorLink, AuthorProfile
from tacme.ui.authors.profiles import get_author_profiles


def _create_author_link_html(
    link: AuthorLink,
) -> str:
    """Render one author profile link."""

    url = escape(
        link.url,
        quote=True,
    )

    label = escape(
        link.label,
    )

    is_external_browser_link = not link.url.startswith(
        (
            "mailto:",
            "tel:",
        )
    )

    external_attributes = ' target="_blank" rel="noopener noreferrer"' if is_external_browser_link else ""

    return (
        '<a class="author-profile-link" '
        f'href="{url}"'
        f"{external_attributes}"
        f' aria-label="{label}">'
        '<span class="author-profile-link-label">'
        f"{label}"
        "</span>"
        "</a>"
    )


def _create_author_visual_html(
    profile: AuthorProfile,
) -> str:
    """Render the optimized photo or fallback monogram."""

    if profile.photo is None:
        return (
            '<div class="author-profile-visual">'
            "<div "
            'class="author-profile-monogram" '
            'aria-hidden="true">'
            f"{escape(profile.initials)}"
            "</div>"
            "</div>"
        )

    avif_source = (
        (f'<source srcset="{escape(profile.photo.avif_url, quote=True)}" type="image/avif">')
        if profile.photo.avif_url is not None
        else ""
    )

    webp_source = (
        (f'<source srcset="{escape(profile.photo.webp_url, quote=True)}" type="image/webp">')
        if profile.photo.webp_url is not None
        else ""
    )

    return (
        '<div class="author-profile-visual is-photo">'
        '<picture class="author-profile-picture">'
        f"{avif_source}"
        f"{webp_source}"
        "<img "
        'class="author-profile-photo" '
        f'src="{escape(profile.photo.jpeg_url, quote=True)}" '
        f'alt="{escape(profile.name, quote=True)}" '
        'width="60" '
        'height="60" '
        'loading="lazy" '
        'decoding="async" '
        'fetchpriority="low">'
        "</picture>"
        "</div>"
    )


def _create_author_card_html(
    profile: AuthorProfile,
    index: int,
) -> str:
    """Render one TACME author card."""

    name = escape(
        profile.name,
    )

    degree_html = (f'<span class="author-profile-degree">{escape(profile.degree)}</span>') if profile.degree else ""

    links_html = "".join(_create_author_link_html(link) for link in profile.links)

    links_block = (f'<div class="author-profile-links">{links_html}</div>') if links_html else ""

    return f"""
    <article
        class="author-profile-card"
        style="--author-card-index: {index};"
    >
        <div class="author-profile-header">
            {_create_author_visual_html(profile)}

            <div class="author-profile-identity">
                <div class="author-profile-title-row">
                    {degree_html}

                    <h3 class="author-profile-name">
                        {name}
                    </h3>
                </div>
            </div>
        </div>

        {links_block}
    </article>
    """


def create_authors_content_html(
    language_index: int,
) -> str:
    """Create the complete localized TACME author-card grid."""

    profiles = get_author_profiles(
        language_index,
    )

    cards = "".join(
        _create_author_card_html(
            profile,
            index,
        )
        for index, profile in enumerate(
            profiles,
        )
    )

    return f"""
    <section class="authors-shell">
        <div class="authors-grid">
            {cards}
        </div>
    </section>
    """
