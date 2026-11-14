"""Public media HTML formatters with backwards-compatible imports."""

from tacme.media.formatting.audio import create_audio_stream_html
from tacme.media.formatting.compatibility import (
    _create_compatibility_tab_state_marker,
    create_media_compatibility_html,
)
from tacme.media.formatting.components import (
    _create_info_row,
    _create_summary_card,
    create_media_unavailable_html,
)
from tacme.media.formatting.overview import create_media_overview_html
from tacme.media.formatting.values import (
    _format_audio_summary,
    _format_bit_rate,
    _format_codec,
    _format_container,
    _format_duration,
    _format_file_size,
    _format_frame_rate,
    _format_optional_value,
    _format_resolution,
    _format_sample_rate,
)
from tacme.media.formatting.video import create_video_stream_html

# Preserve the original public function path for introspection and serialization.
create_audio_stream_html.__module__ = "tacme.media.formatting"
create_media_compatibility_html.__module__ = "tacme.media.formatting"
create_media_overview_html.__module__ = "tacme.media.formatting"
create_media_unavailable_html.__module__ = "tacme.media.formatting"
create_video_stream_html.__module__ = "tacme.media.formatting"

__all__ = [
    "create_audio_stream_html",
    "create_media_compatibility_html",
    "create_media_overview_html",
    "create_media_unavailable_html",
    "create_video_stream_html",
]
