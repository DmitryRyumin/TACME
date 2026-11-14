"""
File: __init__.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Backward-compatible public API for the TACME Authors tab.

License: MIT License
"""

from tacme.ui.authors.contracts import (
    AUTHOR_LINK_LABEL_KEYS as AUTHOR_LINK_LABEL_KEYS,
    AUTHOR_LINK_ORDER as AUTHOR_LINK_ORDER,
    AuthorLink as AuthorLink,
    AuthorPhotoSources as AuthorPhotoSources,
    AuthorProfile as AuthorProfile,
    AuthorsTabComponents as AuthorsTabComponents,
)
from tacme.ui.authors.profiles import (
    _create_author_link_label as _create_author_link_label,
    _create_gradio_file_url as _create_gradio_file_url,
    _get_author_profile as _get_author_profile,
    _resolve_author_photo as _resolve_author_photo,
    get_author_profiles as get_author_profiles,
)
from tacme.ui.authors.rendering import (
    _create_author_card_html as _create_author_card_html,
    _create_author_link_html as _create_author_link_html,
    _create_author_visual_html as _create_author_visual_html,
    create_authors_content_html as create_authors_content_html,
)
from tacme.ui.authors.tab import create_authors_tab as create_authors_tab

# Preserve the historical module location for introspection and serialization.
AuthorLink.__module__ = __name__
AuthorPhotoSources.__module__ = __name__
AuthorProfile.__module__ = __name__
AuthorsTabComponents.__module__ = __name__
_create_author_link_label.__module__ = __name__
_create_gradio_file_url.__module__ = __name__
_resolve_author_photo.__module__ = __name__
_get_author_profile.__module__ = __name__
get_author_profiles.__module__ = __name__
_create_author_link_html.__module__ = __name__
_create_author_visual_html.__module__ = __name__
_create_author_card_html.__module__ = __name__
create_authors_content_html.__module__ = __name__
create_authors_tab.__module__ = __name__
