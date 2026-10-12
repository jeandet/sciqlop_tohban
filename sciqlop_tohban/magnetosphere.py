"""Average magnetopause (MP) and bow shock (BS) of Mercury, and the region a position is in.

Models: Winslow et al. (2013), https://doi.org/10.1002/jgra.50237 — a Shue MP and
a Slavin conic BS, both axisymmetric around the MSO X axis shifted north by the
dipole offset. Positions are BC_MSO km; the classification follows the
Tohban quicklook script: compare the distance to that axis with each boundary's
radius at the same X.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

RM_KM = 2440.0
DIPOLE_Z = 0.2 * RM_KM
# The models are open downstream; cut them where the plots end.
TAIL_X = -10.0 * RM_KM
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Boundary:
    """Conic-like surface: radius(theta) around a focus at (x_focus, DIPOLE_Z)."""

    radius: Callable[[np.ndarray], np.ndarray]
    x_focus: float = 0.0
    # theta beyond which the surface is at infinity (BS hyperbola asymptote).
    theta_max: float = np.pi


def _shue(rss: float, alpha: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda theta: rss * (2.0 / (1.0 + np.cos(theta))) ** alpha


def _conic(semi_latus_rectum: float, eccentricity: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda theta: semi_latus_rectum / (1.0 + eccentricity * np.cos(theta))


MP = Boundary(_shue(1.45 * RM_KM, 0.5))
BS = Boundary(_conic(2.75 * 1.04 * RM_KM, 1.04), x_focus=0.5 * RM_KM, theta_max=np.arccos(-1 / 1.04))


def profile(boundary: Boundary, samples: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """(x, rho) of the boundary, x ascending from TAIL_X to the nose."""
    theta = np.linspace(boundary.theta_max - 1e-5, 0.0, samples)
    radius = boundary.radius(theta)
    x = boundary.x_focus + radius * np.cos(theta)
    keep = x >= TAIL_X
    return x[keep], radius[keep] * np.sin(theta[keep])


def sunward_nose(boundary: Boundary) -> float:
    return float(profile(boundary)[0][-1])


def meridian_curve(boundary: Boundary) -> np.ndarray:
    """(n, 2) X–Z km curve in the noon–midnight meridian, from south tail to north tail."""
    x, rho = profile(boundary)
    return np.column_stack([np.concatenate([x, x[::-1]]), DIPOLE_Z + np.concatenate([-rho, rho[::-1]])])


def _rho(xyz: np.ndarray) -> np.ndarray:
    return np.hypot(xyz[:, 1], xyz[:, 2] - DIPOLE_Z)


def _radius_at(boundary: Boundary, x: np.ndarray) -> np.ndarray:
    """Boundary rho at each X: 0 sunward of the nose, NaN beyond the tail cut."""
    px, prho = profile(boundary)
    return np.interp(x, px, prho, left=np.nan, right=0.0)


def classify(xyz: np.ndarray) -> np.ndarray:
    """Region of each BC_MSO position: "MSP", "MSH", "SW" or "unknown"."""
    xyz = np.asarray(xyz, dtype=float)
    rho, x = _rho(xyz), xyz[:, 0]
    mp, bs = _radius_at(MP, x), _radius_at(BS, x)
    regions = np.where(rho < mp, "MSP", np.where(rho < bs, "MSH", "SW")).astype("U7")
    regions[np.isnan(mp) | np.isnan(bs)] = UNKNOWN
    return regions


def boundary_radius(boundary: Boundary, xyz: np.ndarray) -> np.ndarray:
    """Boundary rho at each position's X, to plot against the position's own rho.

    NaN where the boundary does not reach that X, so a plotted curve breaks there.
    """
    px, prho = profile(boundary)
    return np.interp(np.asarray(xyz, dtype=float)[:, 0], px, prho, left=np.nan, right=np.nan)


def axis_distance(xyz: np.ndarray) -> np.ndarray:
    return _rho(np.asarray(xyz, dtype=float))


def orbit_plane_basis(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """(2, 3) in-plane axes of the orbit at its apoapsis, apoapsis on the negative first axis."""
    apo = int(np.argmax(np.linalg.norm(position, axis=1)))
    normal = np.cross(position[apo], velocity[apo])
    normal /= np.linalg.norm(normal)
    first = -(position[apo] - (position[apo] @ normal) * normal)
    first /= np.linalg.norm(first)
    return np.vstack([first, np.cross(normal, first)])


def plane_section(boundary: Boundary, basis: np.ndarray, rays: int = 720, steps: int = 700) -> np.ndarray:
    """Closed (n, 2) contour where the boundary surface cuts the plane spanned by ``basis``.

    Each in-plane ray from Mercury's centre stops at its first crossing from
    inside to outside the boundary. Rays that leave through the tail cut give
    NaN, so the open side of the contour is not bridged by a straight line.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, rays, endpoint=False)
    directions = np.cos(angles)[:, None] * basis[0] + np.sin(angles)[:, None] * basis[1]
    # Quadratic spacing: fine steps near the planet, where the contour bends most.
    lengths = (np.linspace(0.0, 1.0, steps) ** 2) * 2.0 * -TAIL_X
    points = directions[:, None, :] * lengths[None, :, None]
    flat = points.reshape(-1, 3)
    radius = _radius_at(boundary, flat[:, 0]).reshape(rays, steps)
    beyond_model = np.isnan(radius)
    excess = _rho(flat).reshape(rays, steps) - radius
    outside = (excess > 0) | beyond_model
    out = np.argmax(outside, axis=1)
    ray = np.arange(rays)
    crosses = outside[ray, out] & ~beyond_model[ray, out] & ~outside[:, 0]
    if not crosses.any():
        return np.empty((0, 2))
    # Linear interpolation of the crossing between the last inside and first outside samples.
    before, after = excess[ray, out - 1], excess[ray, out]
    fraction = (before / (before - after))[:, None]
    hits = points[ray, out - 1] + fraction * (points[ray, out] - points[ray, out - 1])
    contour = np.where(crosses[:, None], hits @ basis.T, np.nan)
    return np.vstack([contour, contour[:1]])


def region_runs(time: np.ndarray, regions: np.ndarray) -> list[tuple[np.datetime64, np.datetime64, str]]:
    """(start, stop, region) for each run of equal regions; a run ends where the next starts."""
    if not len(time):
        return []
    starts = np.flatnonzero(np.r_[True, regions[1:] != regions[:-1]])
    stops = np.r_[starts[1:], len(time) - 1]
    return [(time[a], time[b], str(regions[a])) for a, b in zip(starts, stops)]
