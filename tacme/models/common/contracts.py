"""
File: contracts.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Feature and prediction contracts used across TACME model components.

License: MIT License
"""

from collections.abc import Mapping
from typing import NotRequired, TypedDict

from torch import Tensor


class FeatureBatch(TypedDict):
    """Multimodal feature tensors consumed by TACME."""

    features: Mapping[str, Tensor]
    feature_masks: NotRequired[Mapping[str, Tensor]]


class EncodedModality(TypedDict):
    """Encoded modality sequence and its valid-token mask."""

    seq: Tensor
    mask: Tensor


class ForwardAux(TypedDict):
    """Intermediate values retained for flow-matching training losses."""

    fused: dict[str, Tensor]
    base_preds: dict[str, Tensor]
    final_preds: dict[str, Tensor]


ModelOutputs = dict[str, Tensor | None]


ModelEmbeddings = dict[str, Tensor]


FlowLossDetails = dict[str, float | str]
