"""
File: __init__.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Public model interfaces for TACME.

License: MIT License
"""

from tacme.models.fusion import MultiModalFusionModel_v2
from tacme.models.loader import (
    DEFAULT_MODEL_PATH,
    TacmeModelLoadError,
    load_tacme_model,
)

__all__ = [
    "DEFAULT_MODEL_PATH",
    "MultiModalFusionModel_v2",
    "TacmeModelLoadError",
    "load_tacme_model",
]
