"""
File: __init__.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Backward-compatible public interface for the main application UI.

License: MIT License
"""

# Keep historical module-level imports available as well as TACME's UI API.
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Final

import gradio as gr
from gradio.helpers import Examples as GradioExamples

from tacme.config import PROJECT_ROOT, get_config_str, get_config_str_list
from tacme.localization import get_localized_text, get_task_labels
from tacme.ui.application.contracts import ApplicationTabComponents as ApplicationTabComponents
from tacme.ui.application.examples import (
    get_example_video_labels as get_example_video_labels,
    get_existing_example_video_paths as get_existing_example_video_paths,
)
from tacme.ui.application.tab import (
    EXAMPLES_PER_PAGE as EXAMPLES_PER_PAGE,
    create_application_tab as create_application_tab,
)
from tacme.ui.application.title import create_application_title_markdown as create_application_title_markdown

# Preserve the prior import module for callers that serialize or introspect
# the dataclass and public callables. Their definitions remain unchanged.
ApplicationTabComponents.__module__ = __name__
create_application_title_markdown.__module__ = __name__
get_example_video_labels.__module__ = __name__
get_existing_example_video_paths.__module__ = __name__
create_application_tab.__module__ = __name__
