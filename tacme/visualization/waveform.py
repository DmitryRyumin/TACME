"""
File: waveform.py

Author: Elena Ryumina, Alexandr Axyonov, and Dmitry Ryumin

Description: Interactive audio-waveform visualization utilities for TACME.

License: MIT License
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import plotly.graph_objects as go
import torch
from torch.nn import functional as torch_functional
import torchcodec.decoders as torchcodec_decoders

if TYPE_CHECKING:
    from torchcodec.decoders._audio_decoder import AudioDecoder as AudioDecoder
else:
    AudioDecoder = torchcodec_decoders.AudioDecoder

from tacme.localization import get_localized_text

WAVEFORM_SAMPLE_RATE: Final = 16_000
WAVEFORM_NUM_CHANNELS: Final = 1
WAVEFORM_MAX_BINS: Final = 2_400

WAVEFORM_ACCENT: Final = "#F97316"
WAVEFORM_ACCENT_SOFT: Final = "rgba(249, 115, 22, 0.72)"
WAVEFORM_GRID: Final = "rgba(161, 161, 170, 0.09)"
WAVEFORM_BORDER: Final = "rgba(161, 161, 170, 0.16)"
WAVEFORM_ZERO_LINE: Final = "rgba(249, 115, 22, 0.44)"
WAVEFORM_SPIKE: Final = "rgba(249, 115, 22, 0.58)"
WAVEFORM_AXIS: Final = "rgba(228, 228, 231, 0.72)"
WAVEFORM_HOVER_BACKGROUND: Final = "#18181B"


@dataclass(frozen=True, slots=True)
class AudioWaveformEnvelope:
    """Compact min/max envelope used to render one TACME audio waveform."""

    times: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    sample_rate: int
    duration_seconds: float


def _tensor_to_float_tuple(values: torch.Tensor) -> tuple[float, ...]:
    """Convert one one-dimensional tensor into immutable Python floats."""

    return tuple(float(value.item()) for value in values)


def create_audio_waveform_envelope(
    media_path: str,
) -> AudioWaveformEnvelope:
    """Decode the media audio stream and create a compact min/max envelope."""

    path = Path(media_path)

    if not path.is_file():
        msg = f"Media file was not found: {path}"
        raise FileNotFoundError(msg)

    decoder = AudioDecoder(
        str(path),
        sample_rate=WAVEFORM_SAMPLE_RATE,
        num_channels=WAVEFORM_NUM_CHANNELS,
    )

    samples = decoder.get_all_samples()
    waveform = samples.data

    if waveform.ndim != 2:
        msg = f"TorchCodec returned an unexpected waveform shape: {tuple(waveform.shape)}."
        raise ValueError(msg)

    if waveform.shape[-1] <= 0:
        msg = "TorchCodec returned an empty audio waveform."
        raise ValueError(msg)

    sample_rate = int(samples.sample_rate)

    if sample_rate <= 0:
        msg = "TorchCodec returned an invalid audio sample rate."
        raise ValueError(msg)

    mono_waveform = (
        waveform.mean(
            dim=0,
        )
        .detach()
        .to(
            device="cpu",
            dtype=torch.float32,
        )
    )

    mono_waveform = torch.nan_to_num(
        mono_waveform,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    sample_count = int(
        mono_waveform.numel(),
    )

    if sample_count <= 0:
        msg = "Decoded audio waveform does not contain samples."
        raise ValueError(msg)

    bin_count = min(
        WAVEFORM_MAX_BINS,
        sample_count,
    )

    signal = mono_waveform.reshape(
        1,
        1,
        sample_count,
    )

    upper = torch_functional.adaptive_max_pool1d(
        signal,
        bin_count,
    ).reshape(-1)

    lower = -torch_functional.adaptive_max_pool1d(
        -signal,
        bin_count,
    ).reshape(-1)

    duration_seconds = sample_count / sample_rate

    times: tuple[float, ...]

    if bin_count == 1:
        times = (0.0,)
    else:
        denominator = bin_count - 1
        times = tuple(duration_seconds * index / denominator for index in range(bin_count))

    return AudioWaveformEnvelope(
        times=times,
        lower=_tensor_to_float_tuple(
            lower,
        ),
        upper=_tensor_to_float_tuple(
            upper,
        ),
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
    )


def create_audio_waveform_figure(
    waveform: AudioWaveformEnvelope,
    language_index: int,
) -> go.Figure:
    """Create a localized interactive Plotly waveform figure."""

    time_label = get_localized_text(
        "Labels_WAVEFORM_TIME",
        language_index,
    )
    amplitude_label = get_localized_text(
        "Labels_WAVEFORM_AMPLITUDE",
        language_index,
    )
    maximum_label = get_localized_text(
        "Labels_WAVEFORM_MAXIMUM",
        language_index,
    )
    minimum_label = get_localized_text(
        "Labels_WAVEFORM_MINIMUM",
        language_index,
    )
    seconds_unit = get_localized_text(
        "Units_SECONDS",
        language_index,
    )

    peak = max(
        max(
            (abs(value) for value in waveform.lower),
            default=0.0,
        ),
        max(
            (abs(value) for value in waveform.upper),
            default=0.0,
        ),
    )

    amplitude_limit = max(
        peak * 1.08,
        0.05,
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=waveform.times,
            y=waveform.lower,
            mode="lines",
            name=minimum_label,
            line={
                "color": WAVEFORM_ACCENT_SOFT,
                "width": 1.0,
            },
            hoverinfo="skip",
            showlegend=False,
        )
    )

    figure.add_trace(
        go.Scatter(
            x=waveform.times,
            y=waveform.upper,
            mode="lines",
            name=maximum_label,
            line={
                "color": WAVEFORM_ACCENT,
                "width": 1.15,
            },
            fill="tonexty",
            fillgradient={
                "type": "vertical",
                "colorscale": [
                    [
                        0.0,
                        "rgba(249, 115, 22, 0.08)",
                    ],
                    [
                        0.5,
                        "rgba(249, 115, 22, 0.36)",
                    ],
                    [
                        1.0,
                        "rgba(249, 115, 22, 0.08)",
                    ],
                ],
            },
            customdata=waveform.lower,
            hovertemplate=(
                f"<b>{time_label}</b>: %{{x:.3f}} {seconds_unit}<br>"
                f"<b>{minimum_label}</b>: %{{customdata:.4f}}<br>"
                f"<b>{maximum_label}</b>: %{{y:.4f}}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    figure.update_layout(
        template="plotly_dark",
        autosize=True,
        height=300,
        margin={
            "l": 48,
            "r": 10,
            "t": 8,
            "b": 38,
        },
        showlegend=False,
        hovermode="x",
        hoverdistance=-1,
        spikedistance=-1,
        dragmode="zoom",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font={
            "color": WAVEFORM_AXIS,
            "size": 12,
        },
        hoverlabel={
            "bgcolor": WAVEFORM_HOVER_BACKGROUND,
            "bordercolor": WAVEFORM_ACCENT,
            "font": {
                "color": "#FAFAFA",
                "size": 12,
            },
        },
        xaxis={
            "title": {
                "text": f"{time_label}, {seconds_unit}",
                "standoff": 8,
            },
            "range": [
                0.0,
                waveform.duration_seconds,
            ],
            "showgrid": True,
            "gridcolor": WAVEFORM_GRID,
            "zeroline": False,
            "showline": True,
            "linecolor": WAVEFORM_BORDER,
            "ticks": "outside",
            "tickcolor": WAVEFORM_BORDER,
            "showspikes": True,
            "spikemode": "across",
            "spikesnap": "cursor",
            "spikecolor": WAVEFORM_SPIKE,
            "spikedash": "dot",
            "spikethickness": 1,
            "fixedrange": False,
            "rangeslider": {
                "visible": True,
                "thickness": 0.085,
                "bgcolor": "rgba(161, 161, 170, 0.025)",
                "bordercolor": WAVEFORM_BORDER,
                "borderwidth": 1,
            },
        },
        yaxis={
            "title": {
                "text": amplitude_label,
                "standoff": 8,
            },
            "range": [
                -amplitude_limit,
                amplitude_limit,
            ],
            "showgrid": True,
            "gridcolor": WAVEFORM_GRID,
            "zeroline": True,
            "zerolinecolor": WAVEFORM_ZERO_LINE,
            "zerolinewidth": 1.5,
            "showline": True,
            "linecolor": WAVEFORM_BORDER,
            "ticks": "outside",
            "tickcolor": WAVEFORM_BORDER,
            "fixedrange": False,
        },
        uirevision="tacme-audio-waveform",
    )

    return figure


__all__ = [
    "AudioWaveformEnvelope",
    "create_audio_waveform_envelope",
    "create_audio_waveform_figure",
]
