"""Turn plan intervals into per-instrument lane curves for plotting.

Each (instrument, mode) becomes an on/off trace at its own level, like a
logic analyzer: high while the mode is active, low otherwise.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .evt import Interval

INSTRUMENT_ORDER = (
    "DPU", "MPPE_HEPi", "MPPE_HEPe", "MPPE_ENA", "MPPE_MIA", "MPPE_MEA", "MPPE_MSA",
    "MGF_I", "MGF_O", "MDM", "MSASI", "P_EWO_E", "P_EWO_B", "P_EFD", "P_AM2P", "P_SORBET",
)

MODE_COLORS = {
    "BASE": "#9ca3af",
    "HKM": "#2563eb",
    "LM": "#f59e0b",
    "MM": "#ef4444",
    "L_HKM": "#1d4ed8",
    "M_HKM": "#7c3aed",
    "OBS:Apoapsis": "#14b8a6",
    "OBS:Periapsis": "#10b981",
    "OBS:LowReso": "#f97316",
    "OBS:Mag": "#ec4899",
    "OBS:SW": "#06b6d4",
}
FALLBACK_COLOR = "#6b7280"
TRACE_HALF_HEIGHT = 0.35


@dataclass(frozen=True)
class Lane:
    instrument: str
    mode: str
    x: np.ndarray
    y: np.ndarray

    @property
    def color(self) -> str:
        return MODE_COLORS.get(self.mode, FALLBACK_COLOR)


def to_epoch(time: np.datetime64) -> float:
    return float(time.astype("datetime64[s]").astype(np.int64))


def merge_spans(spans: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, stop in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def _mode_key(mode: str) -> tuple[bool, bool, str]:
    return mode != "BASE", mode.startswith("OBS:"), mode


def _instrument_key(instrument: str) -> tuple[int, str]:
    order = INSTRUMENT_ORDER.index(instrument) if instrument in INSTRUMENT_ORDER else len(INSTRUMENT_ORDER)
    return order, instrument


def _is_active(grid: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
    starts, stops = np.array(spans, dtype=float).T
    index = np.searchsorted(starts, grid, side="right") - 1
    return (index >= 0) & (grid < stops[np.maximum(index, 0)])


def _trace(grid: np.ndarray, spans: list[tuple[float, float]], level: float) -> np.ndarray:
    return np.where(_is_active(grid, spans), level + TRACE_HALF_HEIGHT, level - TRACE_HALF_HEIGHT)


def _regular_grid(spans: dict[str, dict[str, list[tuple[float, float]]]], cadence: float) -> np.ndarray:
    bounds = [bound for modes in spans.values() for items in modes.values() for span in items for bound in span]
    return np.arange(min(bounds), max(bounds) + cadence, cadence)


def lanes(intervals: Iterable[Interval], cadence: float = 60.0) -> list[Lane]:
    """One on/off trace per (instrument, mode); modes of an instrument stack top-down.

    simplify: traces are sampled every ``cadence`` seconds because SciQLopPlots
    line graphs break wherever a step is >1.5x its neighbours' (gap detection),
    step line styles included. Edges are therefore shown to ``cadence``
    resolution; exact steps need SciQLop/SciQLopPlots#118.
    """
    spans: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for interval in intervals:
        spans[interval.instrument][interval.mode].append((to_epoch(interval.start), to_epoch(interval.stop)))
    if not spans:
        return []
    grid = _regular_grid(spans, cadence)
    return [
        Lane(instrument, mode, grid, _trace(grid, merge_spans(spans[instrument][mode]), level=-rank))
        for instrument in sorted(spans, key=_instrument_key)
        for rank, mode in enumerate(sorted(spans[instrument], key=_mode_key))
    ]


