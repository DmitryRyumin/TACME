"""
File: media_info.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Media Info updates and modal event bindings.

License: MIT License
"""

from tacme.events.media_info import (
    handle_hide_media_info,
    handle_show_media_info,
)
from tacme.ui.application import ApplicationTabComponents
from tacme.ui.client_scripts import (
    MEDIA_INFO_MODAL_CLOSE_ANIMATION_JS,
    MEDIA_INFO_MODAL_OPEN_ANIMATION_JS,
    MEDIA_INFO_MODAL_PREPARE_OPEN_JS,
)


def register_media_info_modal_events(
    application_content: ApplicationTabComponents,
) -> None:
    """Bind Media Info open, animation and close in their original order."""

    media_info_open_event = application_content.media_info_button.click(
        fn=handle_show_media_info,
        inputs=[],
        outputs=[
            application_content.media_info_modal,
            application_content.media_info_tabs,
        ],
        js=MEDIA_INFO_MODAL_PREPARE_OPEN_JS,
        queue=False,
        show_progress="hidden",
    )

    media_info_open_event.then(
        fn=None,
        inputs=[],
        outputs=[],
        js=MEDIA_INFO_MODAL_OPEN_ANIMATION_JS,
        queue=False,
        show_progress="hidden",
    )

    application_content.media_info_modal_close_button.click(
        fn=handle_hide_media_info,
        inputs=[],
        outputs=[
            application_content.media_info_modal,
        ],
        js=MEDIA_INFO_MODAL_CLOSE_ANIMATION_JS,
        queue=False,
        show_progress="hidden",
    )
