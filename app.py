"""
File: app.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: TACME Gradio application entry point.

License: MIT License
"""

import logging
import os
from typing import cast

import gradio as gr

from tacme.config import (
    PROJECT_ROOT,
    get_config_int,
    get_config_str,
    is_hugging_face_space,
)
from tacme.events.event_handlers import (
    setup_app_event_handlers,
)
from tacme.inference import (
    preload_tacme_inference_engine,
)
from tacme.logger import setup_logging
from tacme.port import ensure_port_available
from tacme.ui.language_selector import (
    create_language_selector,
)
from tacme.ui.tabs import create_app_tabs

LOGGER = logging.getLogger(
    __name__,
)


def create_gradio_app() -> gr.Blocks:
    """Create the TACME Gradio application."""

    app_title = get_config_str(
        "App_TITLE",
        "TACME",
    )

    with (
        gr.Blocks(
            title=app_title,
            fill_width=True,
        ) as gradio_app,
        gr.Column(
            elem_classes="app-shell",
        ),
    ):
        language_selector = create_language_selector()

        app_tabs = create_app_tabs()

        setup_app_event_handlers(
            gradio_app=gradio_app,
            language_selector=language_selector,
            app_tabs=app_tabs,
        )

    return cast(
        gr.Blocks,
        gradio_app,
    )


def main() -> None:
    """Run the TACME Gradio application."""

    configured_server_name = get_config_str(
        "Server_NAME",
        "127.0.0.1",
    )

    configured_server_port = get_config_int(
        "Server_PORT",
        7860,
    )

    default_server_name = "0.0.0.0" if is_hugging_face_space() else configured_server_name

    server_name = os.getenv(
        "GRADIO_SERVER_NAME",
        default_server_name,
    )

    server_port = int(
        os.getenv(
            "PORT",
            os.getenv(
                "GRADIO_SERVER_PORT",
                str(
                    configured_server_port,
                ),
            ),
        )
    )

    app_css_path = get_config_str(
        "App_CSS_PATH",
        "app.css",
    )

    ensure_port_available(
        host=server_name,
        port=server_port,
    )

    LOGGER.info(
        "Loading TACME inference runtime.",
    )

    inference_engine = preload_tacme_inference_engine()

    LOGGER.info(
        "TACME inference runtime loaded on device %s.",
        inference_engine.device,
    )

    static_dir = (PROJECT_ROOT / "static").resolve()

    create_gradio_app().queue(
        api_open=False,
    ).launch(
        theme="default",
        css_paths=app_css_path,
        share=True,
        server_name=server_name,
        server_port=server_port,
        footer_links=[],
        allowed_paths=[
            str(
                static_dir,
            ),
        ],
    )


if __name__ == "__main__":
    setup_logging()
    main()
