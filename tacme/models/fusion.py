"""
File: fusion.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: TACME task-aware directed cross-modal expert fusion models.

License: MIT License
"""

from collections.abc import Iterable, Mapping
from typing import ClassVar, Final, cast

import torch
from torch import Tensor, nn

from tacme.models.common import (
    BaseExpertModel,
    DirectedExpertBlock,
    EncodedModality,
    FlowMatchingMLP,
    TemporalModalityEncoder,
)

DEFAULT_MODALITY_INPUT_DIM: Final[tuple[tuple[str, int], ...]] = (
    ("audio", 1024),
    ("text", 768),
    ("scene", 512),
    ("behavior", 768),
)

DEFAULT_HIDDEN_DIM: Final = 128
DEFAULT_NUM_HEADS: Final = 8
DEFAULT_OUT_DIM: Final = 128
DEFAULT_DROPOUT: Final = 0.15
DEFAULT_TRANSFORMER_LAYERS: Final = 5
DEFAULT_POSITIONAL_ENCODING: Final = False
DEFAULT_HEAD_TYPE: Final = "mlp"
DEFAULT_FLOW_MATCHING_MODE: Final = "heads+features"
DEFAULT_FLOW_MATCHING_STEPS: Final = 4

EMOTION_OUTPUT_DIM: Final = 7
PERSONALITY_OUTPUT_DIM: Final = 5
AH_OUTPUT_DIM: Final = 2


def _resolve_modality_input_dim(
    modality_input_dim: Mapping[str, int] | None,
) -> Mapping[str, int]:
    """Return explicit modality dimensions or the TACME runtime defaults."""

    if modality_input_dim is not None:
        return modality_input_dim

    return dict(
        DEFAULT_MODALITY_INPUT_DIM,
    )


class MultiModalFusionModel_v1(BaseExpertModel):
    """Shared directed experts with uniform expert averaging."""

    def __init__(
        self,
        modality_input_dim: Mapping[str, int] | None = None,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_heads: int = DEFAULT_NUM_HEADS,
        out_dim: int = DEFAULT_OUT_DIM,
        emo_out_dim: int = EMOTION_OUTPUT_DIM,
        pkl_out_dim: int = PERSONALITY_OUTPUT_DIM,
        ah_out_dim: int = AH_OUTPUT_DIM,
        device: str | torch.device = "cpu",
        dropout: float = DEFAULT_DROPOUT,
        ablation_cfg: object | None = None,
        tr_layers: int = DEFAULT_TRANSFORMER_LAYERS,
        positional_encoding: bool = DEFAULT_POSITIONAL_ENCODING,
        head_type: str = DEFAULT_HEAD_TYPE,
        flow_matching_mode: str = DEFAULT_FLOW_MATCHING_MODE,
        flow_matching_steps: int = DEFAULT_FLOW_MATCHING_STEPS,
    ) -> None:
        """Initialize the shared-expert TACME baseline."""

        resolved_modality_input_dim = _resolve_modality_input_dim(
            modality_input_dim,
        )

        super().__init__(
            modality_input_dim=resolved_modality_input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            out_dim=out_dim,
            emo_out_dim=emo_out_dim,
            pkl_out_dim=pkl_out_dim,
            ah_out_dim=ah_out_dim,
            device=device,
            dropout=dropout,
            ablation_cfg=ablation_cfg,
            tr_layers=tr_layers,
            positional_encoding=positional_encoding,
            head_type=head_type,
            flow_matching_mode=flow_matching_mode,
            flow_matching_steps=flow_matching_steps,
        )

        self.modality_encoders = nn.ModuleDict(
            {
                modality: TemporalModalityEncoder(
                    input_dim=input_dim,
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    tr_layers=self.tr_layers,
                    dropout=self.dropout,
                    positional_encoding=self.positional_encoding,
                )
                for modality, input_dim in self.modalities.items()
            }
        )

        self.expert_bank = nn.ModuleDict(
            {
                specification.name: DirectedExpertBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                for specification in self.expert_specs
            }
        )

        if self.flow_matching_mode in {
            "features",
            "heads+features",
        }:
            self.feature_flow_modules = nn.ModuleDict(
                {
                    task: FlowMatchingMLP(
                        state_dim=self.hidden_dim,
                        cond_dim=self.hidden_dim,
                        hidden_dim=self.hidden_dim,
                        dropout=self.dropout,
                    )
                    for task in self.active_tasks
                }
            )

        if self.flow_matching_mode in {
            "heads",
            "heads+features",
        }:
            self.head_flow_modules = nn.ModuleDict(
                {
                    task: FlowMatchingMLP(
                        state_dim=self._task_output_dims[task],
                        cond_dim=self.hidden_dim,
                        hidden_dim=self.hidden_dim,
                        dropout=self.dropout,
                    )
                    for task in self.active_tasks
                }
            )

    def _build_expert_outputs(
        self,
        encoded_modalities: Mapping[
            str,
            EncodedModality,
        ],
    ) -> dict[str, Tensor]:
        """Run the shared directed expert bank."""

        return self._run_expert_bank(
            encoded_modalities,
            self.expert_bank,
        )

    def _combine_experts(
        self,
        task: str,
        expert_outputs: Mapping[str, Tensor],
    ) -> Tensor:
        """Average all available expert representations."""

        del task

        stacked = torch.stack(
            tuple(expert_outputs.values()),
            dim=1,
        )

        return stacked.mean(
            dim=1,
        )

    def _shared_expert_parameters(
        self,
    ) -> Iterable[nn.Parameter]:
        """Return trainable shared directed-expert parameters."""

        return self.expert_bank.parameters()


class MultiModalFusionModel_v2(MultiModalFusionModel_v1):
    """TACME fusion with shared experts and task-aware dynamic expert gates."""

    TASK_KEYS: ClassVar[dict[str, str]] = {
        "emotion": "emo",
        "personality": "pers",
        "ah": "ah",
    }

    def __init__(
        self,
        modality_input_dim: Mapping[str, int] | None = None,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        num_heads: int = DEFAULT_NUM_HEADS,
        out_dim: int = DEFAULT_OUT_DIM,
        emo_out_dim: int = EMOTION_OUTPUT_DIM,
        pkl_out_dim: int = PERSONALITY_OUTPUT_DIM,
        ah_out_dim: int = AH_OUTPUT_DIM,
        device: str | torch.device = "cpu",
        dropout: float = DEFAULT_DROPOUT,
        ablation_cfg: object | None = None,
        tr_layers: int = DEFAULT_TRANSFORMER_LAYERS,
        positional_encoding: bool = DEFAULT_POSITIONAL_ENCODING,
        head_type: str = DEFAULT_HEAD_TYPE,
        flow_matching_mode: str = DEFAULT_FLOW_MATCHING_MODE,
        flow_matching_steps: int = DEFAULT_FLOW_MATCHING_STEPS,
    ) -> None:
        """Initialize the checkpoint-compatible TACME fusion model."""

        super().__init__(
            modality_input_dim=modality_input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            out_dim=out_dim,
            emo_out_dim=emo_out_dim,
            pkl_out_dim=pkl_out_dim,
            ah_out_dim=ah_out_dim,
            device=device,
            dropout=dropout,
            ablation_cfg=ablation_cfg,
            tr_layers=tr_layers,
            positional_encoding=positional_encoding,
            head_type=head_type,
            flow_matching_mode=flow_matching_mode,
            flow_matching_steps=flow_matching_steps,
        )

        self.task_gates = nn.ModuleDict(
            {
                self.TASK_KEYS[task]: nn.Sequential(
                    nn.Linear(
                        self.hidden_dim,
                        self.hidden_dim,
                    ),
                    nn.LayerNorm(
                        self.hidden_dim,
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        self.dropout,
                    ),
                    nn.Linear(
                        self.hidden_dim,
                        1,
                    ),
                )
                for task in self.active_tasks
            }
        )

    def _combine_experts(
        self,
        task: str,
        expert_outputs: Mapping[str, Tensor],
    ) -> Tensor:
        """Fuse available experts with input-dependent task-specific gates."""

        stacked = torch.stack(
            tuple(expert_outputs.values()),
            dim=1,
        )

        gate_key = self.TASK_KEYS[task]

        gate = self.task_gates[gate_key]

        scores = cast(
            Tensor,
            gate(
                stacked,
            ),
        ).squeeze(
            -1,
        )

        weights = torch.softmax(
            scores,
            dim=1,
        )

        return torch.sum(
            stacked
            * weights.unsqueeze(
                -1,
            ),
            dim=1,
        )


__all__ = [
    "MultiModalFusionModel_v1",
    "MultiModalFusionModel_v2",
]
