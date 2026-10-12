"""Shape ephemeris samples into curves for the trajectory and MSASI plots."""

from __future__ import annotations

import numpy as np

from .ephemeris import shell_footprint


def mode_tracks(xy: np.ndarray, modes: np.ndarray) -> dict[str, np.ndarray]:
    """One copy of the track per mode, NaN where another mode is active.

    A sample also belongs to the previous sample's mode, so each run reaches
    the start of the next one and the coloured pieces join up.
    """
    tracks = {}
    for mode in np.unique(modes):
        own = modes == mode
        keep = own | np.r_[False, own[:-1]]
        tracks[str(mode)] = np.where(keep[:, None], xy, np.nan)
    return tracks


def hourly(time: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Positions interpolated at each full hour within the samples."""
    first_hour = (time[0] + np.timedelta64(3599, "s")).astype("datetime64[h]")
    hours = np.arange(first_hour, time[-1] + np.timedelta64(1, "s"), np.timedelta64(1, "h"))
    seconds = (time - time[0]).astype(float)
    at = (hours.astype("datetime64[s]") - time[0]).astype(float)
    return np.column_stack([np.interp(at, seconds, points[:, i]) for i in range(points.shape[1])])


def polylines(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Several (n, 2) curves as one x, y pair, separated by NaN breaks."""
    if not curves:
        return np.empty(0), np.empty(0)
    gap = np.full((1, 2), np.nan)
    joined = np.vstack([part for curve in curves for part in (curve, gap)][:-1])
    return joined[:, 0], joined[:, 1]


def lon_lat(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = np.degrees(np.arctan2(points[..., 1], points[..., 0]))
    lat = np.degrees(np.arcsin(np.clip(points[..., 2] / np.linalg.norm(points, axis=-1), -1.0, 1.0)))
    return lon, lat


def fov_on_shell(origin: np.ndarray, corners: np.ndarray, radius: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices of ``count`` evenly spread samples, and where their FoV corners hit the shell."""
    picked = np.unique(np.linspace(0, len(origin) - 1, count).astype(int))
    hits = shell_footprint(origin[picked][:, None, :], corners[picked], radius)
    return picked, hits
