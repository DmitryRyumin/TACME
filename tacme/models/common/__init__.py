"""
File: __init__.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Backward-compatible public interfaces for TACME shared model components.

License: MIT License
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from typing import ClassVar, NotRequired, TypedDict, cast

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tacme.models.common.ablation import (
    AblationCfg as AblationCfg,
    _get_ablation_value as _get_ablation_value,
    _optional_bool as _optional_bool,
    _optional_string_tuple as _optional_string_tuple,
)
from tacme.models.common.attention import (
    AddAndNorm as AddAndNorm,
    PositionalEncoding as PositionalEncoding,
    PositionWiseFeedForward as PositionWiseFeedForward,
    TransformerEncoderLayer as TransformerEncoderLayer,
)
from tacme.models.common.base import (
    SUPPORTED_FLOW_MATCHING_MODES as SUPPORTED_FLOW_MATCHING_MODES,
    BaseExpertModel as BaseExpertModel,
)
from tacme.models.common.contracts import (
    EncodedModality as EncodedModality,
    FeatureBatch as FeatureBatch,
    FlowLossDetails as FlowLossDetails,
    ForwardAux as ForwardAux,
    ModelEmbeddings as ModelEmbeddings,
    ModelOutputs as ModelOutputs,
)
from tacme.models.common.encoders import TemporalModalityEncoder as TemporalModalityEncoder
from tacme.models.common.experts import DirectedExpertBlock as DirectedExpertBlock, ExpertSpec as ExpertSpec
from tacme.models.common.flow_matching import FlowMatchingMLP as FlowMatchingMLP
from tacme.models.common.heads import SUPPORTED_HEAD_TYPES as SUPPORTED_HEAD_TYPES, TaskHead as TaskHead
from tacme.models.common.tensor_ops import (
    ensure_sequence_input as ensure_sequence_input,
    masked_mean as masked_mean,
    module_device as module_device,
    resolve_device as resolve_device,
)

# Keep the original qualified class paths for legacy PyTorch/pickle consumers.
# The implementations themselves are unchanged and live in thematic modules.
for _legacy_class in (
    FeatureBatch,
    EncodedModality,
    ForwardAux,
    PositionWiseFeedForward,
    AddAndNorm,
    PositionalEncoding,
    TransformerEncoderLayer,
    TemporalModalityEncoder,
    TaskHead,
    FlowMatchingMLP,
    DirectedExpertBlock,
    ExpertSpec,
    AblationCfg,
    BaseExpertModel,
):
    _legacy_class.__module__ = __name__
del _legacy_class

__all__ = [
    "AblationCfg",
    "BaseExpertModel",
    "DirectedExpertBlock",
    "EncodedModality",
    "ExpertSpec",
    "FeatureBatch",
    "FlowMatchingMLP",
    "ModelEmbeddings",
    "ModelOutputs",
    "TaskHead",
    "TemporalModalityEncoder",
    "TransformerEncoderLayer",
]
