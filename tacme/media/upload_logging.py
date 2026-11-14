"""Avoid noisy Uvicorn tracebacks for aborted Gradio file uploads.

An interrupted multipart upload is a routine client disconnect. This filter
only suppresses the *log record* for that exact situation; it never handles,
changes, or swallows the request exception itself. Every other Uvicorn error
remains visible, including connection failures during inference.
"""

import logging
from traceback import extract_tb


class _UploadClientDisconnectFilter(logging.Filter):
    """Hide only Gradio upload-route ClientDisconnect exception tracebacks."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Leave unrelated errors untouched, including other disconnects."""
        if (
            record.name != "uvicorn.error"
            or record.getMessage().strip() != "Exception in ASGI application"
            or not record.exc_info
            or type(record.exc_info[1]).__module__ != "starlette.requests"
            or type(record.exc_info[1]).__name__ != "ClientDisconnect"
            or record.exc_info[2] is None
        ):
            return True

        # A disconnect outside the upload endpoint (e.g. analysis streaming)
        # remains a normal error. Do not filter by exception class alone.
        upload_route_found = any(
            frame.name == "upload_file" and frame.filename.replace("\\", "/").endswith("/gradio/routes.py")
            for frame in extract_tb(record.exc_info[2])
        )
        return not upload_route_found


def install_upload_disconnect_log_filter() -> None:
    """Install the narrow filter once, before Gradio starts serving requests."""
    logger = logging.getLogger("uvicorn.error")
    if not any(isinstance(item, _UploadClientDisconnectFilter) for item in logger.filters):
        logger.addFilter(_UploadClientDisconnectFilter())
