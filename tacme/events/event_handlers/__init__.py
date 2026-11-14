"""
File: __init__.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Backward-compatible entry point for TACME Gradio event registration.

License: MIT License
"""

from tacme.events.event_handlers.registration import (
    setup_app_event_handlers as setup_app_event_handlers,
)

__all__ = ["setup_app_event_handlers"]
