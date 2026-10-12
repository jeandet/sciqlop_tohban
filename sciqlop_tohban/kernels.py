"""Fetch the public ESA BepiColombo planning kernels Tohban needs, and only those.

ESA's ``bc_plan.tm`` loads ~500 MB. MIO/MPO positions and MSASI pointing need
~150 MB of it. Kernel file names carry their version, so a cached file never
goes stale: a newer plan names a new file.
"""

from __future__ import annotations

import os
import re
import urllib.request
from pathlib import Path
from typing import Callable

ESA_KERNELS = "https://spiftp.esac.esa.int/data/SPICE/BEPICOLOMBO/kernels/"
PLAN_METAKERNEL = "mk/bc_plan.tm"
CACHE = Path.home() / ".cache" / "sciqlop_tohban" / "kernels"
# Set to a metakernel path to use your own kernel set instead.
METAKERNEL_ENV = "TOHBAN_METAKERNEL"

_WANTED = re.compile(
    r"^(lsk/"
    r"|pck/.*\.tpc$"
    r"|fk/bc_"
    r"|ik/bc_mmo_msasi"
    r"|sclk/bc_mmo"
    r"|ck/bc_mmo_sc_"
    r"|spk/(de\d+s|bc_sci|bc_mmo_(struct|cruise|mlt)|bc_mpo_(mlt|fcp)))"
)
_LISTED = re.compile(r"'\$KERNELS/([^']+)'")

Fetch = Callable[[str], bytes]


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def wanted_kernels(metakernel_text: str) -> list[str]:
    return [name for name in _LISTED.findall(metakernel_text) if _WANTED.match(name)]


def local_metakernel(kernels: list[str], folder: Path) -> str:
    # A path symbol, as ESA does: text kernel lines are limited to 132 characters.
    listed = "\n".join(f"    '$KERNELS/{name}'" for name in kernels)
    return (
        "\\begindata\n"
        f"PATH_VALUES = ( '{folder}' )\n"
        "PATH_SYMBOLS = ( 'KERNELS' )\n"
        f"KERNELS_TO_LOAD = (\n{listed}\n)\n"
        "\\begintext\n"
    )


def _fetch_missing(kernels: list[str], folder: Path, fetch: Fetch) -> None:
    for name in kernels:
        target = folder / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            partial.write_bytes(fetch(ESA_KERNELS + name))
            partial.rename(target)


def ensure_kernels(folder: Path = CACHE, fetch: Fetch = _download) -> Path:
    """Path of a metakernel loading the cached kernels, downloading what is missing."""
    metakernel = folder / "tohban.tm"
    try:
        plan = fetch(ESA_KERNELS + PLAN_METAKERNEL).decode("ascii")
    except OSError:
        if metakernel.exists():
            return metakernel
        raise
    kernels = wanted_kernels(plan)
    folder.mkdir(parents=True, exist_ok=True)
    _fetch_missing(kernels, folder, fetch)
    metakernel.write_text(local_metakernel(kernels, folder))
    return metakernel


def metakernel() -> Path:
    override = os.environ.get(METAKERNEL_ENV)
    return Path(override) if override else ensure_kernels()
