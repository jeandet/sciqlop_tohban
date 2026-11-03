"""MIO/MPO positions and MSASI pointing sampled from SPICE, in BC_MSO km."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterator

import numpy as np

STEP = np.timedelta64(5, "m")
MSASI_FOV_ID = -68401
FRAME = "BC_MSO"
OBSERVER = "MERCURY"


class MissingEphemeris(RuntimeError):
    """SPICE has no coverage over the requested window."""


@contextmanager
def coverage(what: str, start: np.datetime64, stop: np.datetime64) -> Iterator[None]:
    from spiceypy.utils.exceptions import SpiceyError

    try:
        yield
    except SpiceyError as error:
        # spiceypy keeps the readable one-liner in 'long' and a banner in str().
        raise MissingEphemeris(f"no {what} between {start} and {stop} in ESA's current plan ({error.long})") from error


@dataclass(frozen=True)
class Ephemeris:
    time: np.ndarray  # datetime64[s]
    mio: np.ndarray  # (n, 3)
    mio_velocity: np.ndarray  # (n, 3) km/s
    mpo: np.ndarray  # (n, 3)
    msasi_boresight: np.ndarray  # (n, 3) unit vectors
    msasi_corners: np.ndarray  # (n, corners, 3) unit vectors

    def window(self, start: np.datetime64, stop: np.datetime64) -> Ephemeris:
        """Samples within [start, stop] plus one on each side, so curves reach the edges."""
        first = max(int(np.searchsorted(self.time, start, side="right")) - 1, 0)
        last = int(np.searchsorted(self.time, stop, side="left")) + 1
        return Ephemeris(*(getattr(self, f.name)[first:last] for f in fields(self)))

    @property
    def mio_radius(self) -> np.ndarray:
        return np.linalg.norm(self.mio, axis=1)

    @property
    def mpo_radius(self) -> np.ndarray:
        return np.linalg.norm(self.mpo, axis=1)

    @property
    def separation(self) -> np.ndarray:
        return np.linalg.norm(self.mio - self.mpo, axis=1)


def shell_footprint(origin: np.ndarray, direction: np.ndarray, radius: float) -> np.ndarray:
    """Nearest forward hit of each ray on a sphere centred on Mercury; NaN when it misses."""
    unit = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
    along = np.sum(origin * unit, axis=-1)
    discriminant = along**2 - (np.sum(origin * origin, axis=-1) - radius**2)
    root = np.sqrt(np.where(discriminant >= 0, discriminant, np.nan))
    near, far = -along - root, -along + root
    distance = np.where(near > 0, near, np.where(far > 0, far, np.nan))
    return origin + distance[..., None] * unit


def _instants(start: np.datetime64, stop: np.datetime64, step: np.timedelta64) -> np.ndarray:
    return np.arange(start, stop + step, step).astype("datetime64[s]")


def sample(metakernel: Path, start: np.datetime64, stop: np.datetime64, step: np.timedelta64 = STEP) -> Ephemeris:
    import spiceypy as spice

    time = _instants(start, stop, step)
    spice.furnsh(str(metakernel))
    try:
        # simplify: one str2et, then UTC offsets; off by a second across a leap second.
        et = spice.str2et(str(time[0])) + (time - time[0]).astype(float)
        with coverage("MIO ephemeris", start, stop):
            mio_state = np.asarray(spice.spkezr("MMO", et, FRAME, "NONE", OBSERVER)[0])
        with coverage("MPO ephemeris", start, stop):
            mpo = np.asarray(spice.spkpos("MPO", et, FRAME, "NONE", OBSERVER)[0])
        _, fov_frame, boresight, _, corners = spice.getfov(MSASI_FOV_ID, 10)
        with coverage("MSASI pointing", start, stop):
            rotation = np.asarray([spice.pxform(fov_frame, FRAME, t) for t in et])
    finally:
        spice.kclear()
    return Ephemeris(
        time=time,
        mio=mio_state[:, :3],
        mio_velocity=mio_state[:, 3:],
        mpo=mpo,
        msasi_boresight=rotation @ (boresight / np.linalg.norm(boresight)),
        msasi_corners=np.einsum("nij,kj->nki", rotation, corners / np.linalg.norm(corners, axis=1, keepdims=True)),
    )
