"""
File: base.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Unmodified BaseExpertModel implementation and flow-matching logic.

License: MIT License
"""

from collections.abc import Iterable, Mapping
from typing import ClassVar, cast

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from tacme.models.common.ablation import AblationCfg
from tacme.models.common.contracts import (
    EncodedModality,
    FeatureBatch,
    FlowLossDetails,
    ForwardAux,
    ModelEmbeddings,
    ModelOutputs,
)
from tacme.models.common.encoders import TemporalModalityEncoder
from tacme.models.common.experts import DirectedExpertBlock, ExpertSpec
from tacme.models.common.flow_matching import FlowMatchingMLP
from tacme.models.common.heads import TaskHead
from tacme.models.common.tensor_ops import module_device, resolve_device

SUPPORTED_FLOW_MATCHING_MODES = frozenset(
    {
        "off",
        "heads",
        "features",
        "heads+features",
    }
)


class BaseExpertModel(nn.Module):
    """Shared implementation for TACME directed-expert fusion models."""

    TASK_OUTPUTS: ClassVar[
        dict[
            str,
            tuple[
                str,
                int,
                bool,
            ],
        ]
    ] = {
        "emotion": (
            "emotion_logits",
            7,
            False,
        ),
        "personality": (
            "personality_scores",
            5,
            True,
        ),
        "ah": (
            "ah_logits",
            2,
            False,
        ),
    }

    def __init__(
        self,
        modality_input_dim: Mapping[str, int],
        hidden_dim: int = 256,
        num_heads: int = 8,
        out_dim: int = 256,
        emo_out_dim: int = 7,
        pkl_out_dim: int = 5,
        ah_out_dim: int = 2,
        device: str | torch.device = "cpu",
        dropout: float = 0.1,
        ablation_cfg: object | None = None,
        tr_layers: int = 5,
        positional_encoding: bool = True,
        head_type: str = "mlp",
        flow_matching_mode: str = "off",
        flow_matching_steps: int = 4,
    ) -> None:
        super().__init__()

        self.device = resolve_device(
            device,
        )

        self.hidden_dim = int(
            hidden_dim,
        )

        self.num_heads = int(
            num_heads,
        )

        self.out_dim = int(
            out_dim,
        )

        self.dropout = float(
            dropout,
        )

        self.tr_layers = int(
            tr_layers,
        )

        self.positional_encoding = bool(
            positional_encoding,
        )

        self.head_type = head_type

        self.flow_matching_mode = flow_matching_mode.lower()

        self.flow_matching_steps = max(
            int(
                flow_matching_steps,
            ),
            1,
        )

        self._validate_hyperparameters()

        normalized_dims = {
            name: int(
                dimension,
            )
            for name, dimension in (modality_input_dim.items())
        }

        if not normalized_dims:
            raise ValueError("At least one modality is required.")

        invalid_dimensions = {name: dimension for name, dimension in (normalized_dims.items()) if dimension <= 0}

        if invalid_dimensions:
            msg = f"Modality dimensions must be positive: {invalid_dimensions}."
            raise ValueError(msg)

        ablation = AblationCfg.from_obj(
            ablation_cfg,
        )

        self.modalities = self._select_modalities(
            normalized_dims,
            ablation,
        )

        self.expert_specs = self._create_expert_specs()

        self.active_tasks = self._select_tasks(
            ablation,
        )

        task_dims = {
            "emotion": int(
                emo_out_dim,
            ),
            "personality": int(
                pkl_out_dim,
            ),
            "ah": int(
                ah_out_dim,
            ),
        }

        if any(dimension <= 0 for dimension in (task_dims.values())):
            msg = f"Task output dimensions must be positive: {task_dims}."
            raise ValueError(msg)

        self.task_heads = nn.ModuleDict(
            {
                task: TaskHead(
                    input_dim=(self.hidden_dim),
                    hidden_dim=(self.out_dim),
                    out_dim=(task_dims[task]),
                    dropout=(self.dropout),
                    head_type=(self.head_type),
                    apply_sigmoid=(task == "personality"),
                )
                for task in (self.active_tasks)
            }
        )

        self.modality_encoders = nn.ModuleDict()

        self.feature_flow_modules = nn.ModuleDict()

        self.head_flow_modules = nn.ModuleDict()

        self._task_output_dims = task_dims

        self._last_forward_aux: ForwardAux | None = None

    def _validate_hyperparameters(
        self,
    ) -> None:
        """Validate model hyperparameters before modules are constructed."""

        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        if self.hidden_dim % self.num_heads != 0:
            msg = f"hidden_dim must be divisible by num_heads; got {self.hidden_dim} and {self.num_heads}."
            raise ValueError(msg)

        if self.out_dim <= 0:
            raise ValueError("out_dim must be positive.")

        if self.tr_layers <= 0:
            raise ValueError("tr_layers must be positive.")

        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in the range [0, 1).")

        if self.flow_matching_mode not in SUPPORTED_FLOW_MATCHING_MODES:
            modes = ", ".join(sorted(SUPPORTED_FLOW_MATCHING_MODES))

            msg = f"flow_matching_mode must be one of: {modes}."
            raise ValueError(msg)

    @staticmethod
    def _select_modalities(
        modality_input_dim: Mapping[str, int],
        ablation: AblationCfg,
    ) -> dict[str, int]:
        """Apply modality ablations while preserving input order."""

        known_modalities = set(
            modality_input_dim,
        )

        disabled = set(
            ablation.disabled_modalities or (),
        )

        active = set(
            ablation.active_modalities or (),
        )

        unknown_modalities = (disabled | active) - known_modalities

        if unknown_modalities:
            bad_fields: dict[
                str,
                list[str],
            ] = {}

            unknown_disabled = sorted(disabled - known_modalities)

            unknown_active = sorted(active - known_modalities)

            if unknown_disabled:
                bad_fields["disabled_modalities"] = unknown_disabled

            if unknown_active:
                bad_fields["active_modalities"] = unknown_active

            msg = f"Unknown modalities in ablation config: {bad_fields}; known: {sorted(known_modalities)}."
            raise KeyError(msg)

        if active:
            modalities = {name: dimension for name, dimension in (modality_input_dim.items()) if name in active}

        else:
            modalities = {name: dimension for name, dimension in (modality_input_dim.items()) if name not in disabled}

        if not modalities:
            raise ValueError("All modalities are disabled; nothing to process.")

        return modalities

    def _create_expert_specs(
        self,
    ) -> tuple[
        ExpertSpec,
        ...,
    ]:
        """Create the stable directed-expert specification sequence."""

        modality_names = tuple(
            self.modalities,
        )

        if (
            len(
                modality_names,
            )
            == 1
        ):
            name = modality_names[0]

            return (
                ExpertSpec(
                    name,
                    name,
                ),
            )

        return tuple(
            ExpertSpec(
                query_mod,
                context_mod,
            )
            for query_mod in (modality_names)
            for context_mod in (modality_names)
            if query_mod != context_mod
        )

    def _select_tasks(
        self,
        ablation: AblationCfg,
    ) -> tuple[
        str,
        ...,
    ]:
        """Validate and return the active task sequence."""

        allowed_tasks = tuple(
            self.TASK_OUTPUTS,
        )

        if ablation.active_tasks is None:
            return allowed_tasks

        unknown_tasks = set(
            ablation.active_tasks,
        ) - set(
            allowed_tasks,
        )

        if unknown_tasks:
            msg = f"Unknown active tasks: {sorted(unknown_tasks)}; known: {sorted(allowed_tasks)}."
            raise KeyError(msg)

        return tuple(
            ablation.active_tasks,
        )

    def _get_modality_tensor(
        self,
        batch: FeatureBatch,
        modality: str,
    ) -> tuple[
        Tensor | None,
        Tensor | None,
    ]:
        """Fetch one modality tensor and optional mask on the model device."""

        features = batch["features"]

        masks = batch.get(
            "feature_masks",
            {},
        )

        tensor = features.get(
            modality,
        )

        if tensor is None:
            return (
                None,
                None,
            )

        device = module_device(
            self,
        )

        tensor = tensor.to(
            device,
        )

        mask = masks.get(
            modality,
        )

        if mask is not None:
            mask = mask.to(
                device,
            )

        return (
            tensor,
            mask,
        )

    def _encode_modalities(
        self,
        batch: FeatureBatch,
    ) -> dict[
        str,
        EncodedModality,
    ]:
        """Encode all modalities present in the current batch."""

        encoded: dict[
            str,
            EncodedModality,
        ] = {}

        for modality in self.modalities:
            tensor, mask = self._get_modality_tensor(
                batch,
                modality,
            )

            if tensor is None:
                continue

            encoder = cast(
                TemporalModalityEncoder,
                self.modality_encoders[modality],
            )

            (
                sequence,
                sequence_mask,
            ) = encoder(
                tensor,
                mask,
            )

            encoded[modality] = {
                "seq": sequence,
                "mask": sequence_mask,
            }

        return encoded

    def _run_expert_bank(
        self,
        encoded_modalities: Mapping[
            str,
            EncodedModality,
        ],
        expert_bank: nn.ModuleDict,
    ) -> dict[
        str,
        Tensor,
    ]:
        """Run all directed experts whose two modalities are available."""

        expert_outputs: dict[
            str,
            Tensor,
        ] = {}

        for spec in self.expert_specs:
            query = encoded_modalities.get(
                spec.query_mod,
            )

            context = encoded_modalities.get(
                spec.context_mod,
            )

            if query is None or context is None:
                continue

            expert = cast(
                DirectedExpertBlock,
                expert_bank[spec.name],
            )

            expert_outputs[spec.name] = expert(
                query["seq"],
                context["seq"],
                query_mask=(query["mask"]),
                context_mask=(context["mask"]),
            )

        return expert_outputs

    def _combine_experts(
        self,
        task: str,
        expert_outputs: Mapping[
            str,
            Tensor,
        ],
    ) -> Tensor:
        """Combine directed expert outputs for one task."""

        raise NotImplementedError

    def _build_expert_outputs(
        self,
        encoded_modalities: Mapping[
            str,
            EncodedModality,
        ],
    ) -> dict[
        str,
        Tensor,
    ]:
        """Build directed expert outputs for a batch."""

        raise NotImplementedError

    def _forward_impl(
        self,
        batch: FeatureBatch,
    ) -> tuple[
        ModelEmbeddings,
        ModelOutputs,
    ]:
        """Execute the shared TACME forward pass."""

        encoded_modalities = self._encode_modalities(
            batch,
        )

        if not encoded_modalities:
            raise ValueError("No valid modalities were provided.")

        expert_outputs = self._build_expert_outputs(
            encoded_modalities,
        )

        if not expert_outputs:
            raise ValueError("No experts produced outputs for the current batch.")

        outputs: ModelOutputs = {
            "emotion_logits": None,
            "personality_scores": None,
            "ah_logits": None,
        }

        embeddings: ModelEmbeddings = {}

        auxiliary: ForwardAux = {
            "fused": {},
            "base_preds": {},
            "final_preds": {},
        }

        for task in self.active_tasks:
            fused = self._combine_experts(
                task,
                expert_outputs,
            )

            auxiliary["fused"][task] = fused

            if self.flow_matching_mode in {
                "features",
                "heads+features",
            }:
                fused = self._integrate_feature_flow(
                    task,
                    fused,
                )

            task_head = cast(
                TaskHead,
                self.task_heads[task],
            )

            (
                embedding,
                prediction,
            ) = task_head.forward_with_embeddings(
                fused,
            )

            auxiliary["base_preds"][task] = prediction

            if self.flow_matching_mode in {
                "heads",
                "heads+features",
            }:
                prediction = self._integrate_head_flow(
                    task,
                    auxiliary["fused"][task],
                    prediction,
                )

            auxiliary["final_preds"][task] = prediction

            embeddings[task] = embedding

            output_name = self.TASK_OUTPUTS[task][0]

            outputs[output_name] = prediction

        self._last_forward_aux = auxiliary

        return (
            embeddings,
            outputs,
        )

    def forward(
        self,
        batch: FeatureBatch,
    ) -> ModelOutputs:
        """Return TACME task predictions."""

        _, outputs = self._forward_impl(
            batch,
        )

        return outputs

    def forward_with_embeddings(
        self,
        batch: FeatureBatch,
    ) -> tuple[
        ModelEmbeddings,
        ModelOutputs,
    ]:
        """Return TACME task embeddings and predictions."""

        return self._forward_impl(
            batch,
        )

    def shared_parameters_for_gradnorm(
        self,
    ) -> list[nn.Parameter]:
        """Return trainable parameters shared across tasks."""

        parameters = [parameter for parameter in (self.modality_encoders.parameters()) if parameter.requires_grad]

        parameters.extend(parameter for parameter in (self._shared_expert_parameters()) if parameter.requires_grad)

        return parameters

    def _shared_expert_parameters(
        self,
    ) -> Iterable[nn.Parameter]:
        """Return parameters shared by task prediction branches."""

        return ()

    def _integrate_flow(
        self,
        module: FlowMatchingMLP,
        start: Tensor,
        condition: Tensor,
    ) -> Tensor:
        """Integrate a learned velocity field with fixed-step Euler updates."""

        x = start

        step = 1.0 / self.flow_matching_steps

        for index in range(self.flow_matching_steps):
            t = torch.full(
                (
                    x.shape[0],
                    1,
                ),
                (index + 0.5) * step,
                device=x.device,
                dtype=x.dtype,
            )

            velocity = module(
                x,
                condition,
                t,
            )

            x = x + step * velocity

        return x

    def _integrate_feature_flow(
        self,
        task: str,
        fused: Tensor,
    ) -> Tensor:
        """Refine a fused representation through feature-space flow."""

        if task not in self.feature_flow_modules:
            return fused

        module = cast(
            FlowMatchingMLP,
            self.feature_flow_modules[task],
        )

        return self._integrate_flow(
            module,
            torch.zeros_like(
                fused,
            ),
            fused,
        )

    def _integrate_head_flow(
        self,
        task: str,
        fused: Tensor,
        base_prediction: Tensor,
    ) -> Tensor:
        """Refine a task prediction through output-space flow."""

        if task not in self.head_flow_modules:
            return base_prediction

        module = cast(
            FlowMatchingMLP,
            self.head_flow_modules[task],
        )

        return self._integrate_flow(
            module,
            base_prediction,
            fused,
        )

    def _task_targets_for_flow(
        self,
        task: str,
        labels: Mapping[
            str,
            Tensor,
        ],
        reference: Tensor,
    ) -> tuple[
        Tensor | None,
        Tensor | None,
    ]:
        """Build a flow-matching target and validity mask for one task."""

        if task == "emotion":
            target = labels.get(
                "emotion",
            )

            mask = labels.get(
                "valid_emo",
            )

            if target is None:
                return (
                    None,
                    None,
                )

            if target.ndim > 1 and target.shape[-1] == self._task_output_dims[task]:
                encoded_target = target.float()

            else:
                encoded_target = F.one_hot(
                    target.long(),
                    num_classes=(self._task_output_dims[task]),
                ).float()

            return (
                encoded_target.to(
                    reference.device,
                ),
                mask,
            )

        if task == "personality":
            target = labels.get(
                "personality",
            )

            mask = labels.get(
                "valid_per",
            )

            if target is None:
                return (
                    None,
                    None,
                )

            return (
                target.float().to(
                    reference.device,
                ),
                mask,
            )

        if task == "ah":
            target = labels.get(
                "ah",
            )

            mask = labels.get(
                "valid_ah",
            )

            if target is None:
                return (
                    None,
                    None,
                )

            if mask is not None:
                mask = mask.to(
                    device=target.device,
                    dtype=torch.bool,
                )

                target = torch.where(
                    mask,
                    target,
                    torch.zeros_like(
                        target,
                    ),
                )

            encoded_target = F.one_hot(
                target.long(),
                num_classes=(self._task_output_dims[task]),
            ).float()

            return (
                encoded_target.to(
                    reference.device,
                ),
                mask,
            )

        return (
            None,
            None,
        )

    def compute_flow_matching_loss(
        self,
        labels: Mapping[
            str,
            Tensor,
        ],
    ) -> tuple[
        Tensor,
        FlowLossDetails,
    ]:
        """Compute the auxiliary TACME flow-matching objective."""

        device = module_device(
            self,
        )

        auxiliary = self._last_forward_aux

        if self.flow_matching_mode == "off" or auxiliary is None:
            return (
                torch.zeros(
                    (),
                    device=device,
                    requires_grad=True,
                ),
                {},
            )

        total: Tensor | None = None

        details: FlowLossDetails = {}

        count = 0

        if self.flow_matching_mode in {
            "features",
            "heads+features",
        }:
            for (
                task,
                fused,
            ) in auxiliary["fused"].items():
                feature_target = fused.detach()

                x0 = torch.randn_like(
                    feature_target,
                )

                t = torch.rand(
                    (
                        feature_target.shape[0],
                        1,
                    ),
                    device=feature_target.device,
                    dtype=feature_target.dtype,
                )

                x_t = (1.0 - t) * x0 + t * feature_target

                target_velocity = feature_target - x0

                module = cast(
                    FlowMatchingMLP,
                    self.feature_flow_modules[task],
                )

                predicted_velocity = module(
                    x_t,
                    fused,
                    t,
                )

                loss = F.mse_loss(
                    predicted_velocity,
                    target_velocity,
                )

                total = loss if total is None else total + loss

                details[f"fm_{task}"] = float(loss.detach().item())

                count += 1

        if self.flow_matching_mode in {
            "heads",
            "heads+features",
        }:
            for (
                task,
                base_prediction,
            ) in auxiliary["base_preds"].items():
                (
                    head_target,
                    valid_mask,
                ) = self._task_targets_for_flow(
                    task,
                    labels,
                    base_prediction,
                )

                if head_target is None:
                    continue

                if valid_mask is not None:
                    valid_mask = valid_mask.to(
                        device=(base_prediction.device),
                        dtype=torch.bool,
                    )

                    if not bool(valid_mask.any()):
                        continue

                    condition = auxiliary["fused"][task][valid_mask]

                    x0 = base_prediction.detach()[valid_mask]

                    x1 = head_target[valid_mask]

                else:
                    condition = auxiliary["fused"][task]

                    x0 = base_prediction.detach()

                    x1 = head_target

                t = torch.rand(
                    (
                        x1.shape[0],
                        1,
                    ),
                    device=x1.device,
                    dtype=x1.dtype,
                )

                x_t = (1.0 - t) * x0 + t * x1

                target_velocity = x1 - x0

                module = cast(
                    FlowMatchingMLP,
                    self.head_flow_modules[task],
                )

                predicted_velocity = module(
                    x_t,
                    condition,
                    t,
                )

                loss = F.mse_loss(
                    predicted_velocity,
                    target_velocity,
                )

                total = loss if total is None else total + loss

                details[f"fm_{task}"] = float(loss.detach().item())

                count += 1

        if total is None or count == 0:
            return (
                torch.zeros(
                    (),
                    device=device,
                    requires_grad=True,
                ),
                {},
            )

        total = total / count

        details["flow_matching_mode"] = self.flow_matching_mode

        return (
            total,
            details,
        )
