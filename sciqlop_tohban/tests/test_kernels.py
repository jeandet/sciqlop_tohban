from pathlib import Path

from sciqlop_tohban.kernels import ensure_kernels, local_metakernel, wanted_kernels

ESA_MK = """\\begindata
     PATH_VALUES       = ( '..' )
     PATH_SYMBOLS      = ( 'KERNELS' )
     KERNELS_TO_LOAD   = (
                           '$KERNELS/ck/bc_mpo_sc_slt_00233_20270309_20291231_f20181127_v01.bc'
                           '$KERNELS/ck/bc_mmo_sc_slt_50038_20261207_20290220_f20231129_v01.bc'
                           '$KERNELS/fk/bc_mmo_v14.tf'
                           '$KERNELS/fk/earth_topo_201023.tf'
                           '$KERNELS/dsk/bc_mmo_sc_bus_v02.bds'
                           '$KERNELS/ik/bc_mmo_msasi_v03.ti'
                           '$KERNELS/ik/bc_mmo_mppe_v04.ti'
                           '$KERNELS/lsk/naif0012.tls'
                           '$KERNELS/pck/pck00011_bc_v00.tpc'
                           '$KERNELS/pck/earth_070425_370426_predict.bpc'
                           '$KERNELS/sclk/bc_mmo_step_20240103.tsc'
                           '$KERNELS/sclk/bc_mpo_step_20261011.tsc'
                           '$KERNELS/spk/de432s.bsp'
                           '$KERNELS/spk/bc_mmo_mlt_50038_20261207_20290220_v01.bsp'
                           '$KERNELS/spk/bc_mmo_slt_extension_20290220_20310220_v01.bsp'
                           '$KERNELS/spk/bc_mpo_mlt_50037_20270301_20290516_v01.bsp'
                           '$KERNELS/spk/bc_mpo_fcp_00237_20181020_20270411_v02.bsp'
                           '$KERNELS/spk/bc_mpo_flp_00233_20270309_20300101_v02.bsp'
                         )
\\begintext
"""

WANTED = [
    "ck/bc_mmo_sc_slt_50038_20261207_20290220_f20231129_v01.bc",
    "fk/bc_mmo_v14.tf",
    "ik/bc_mmo_msasi_v03.ti",
    "lsk/naif0012.tls",
    "pck/pck00011_bc_v00.tpc",
    "sclk/bc_mmo_step_20240103.tsc",
    "spk/de432s.bsp",
    "spk/bc_mmo_mlt_50038_20261207_20290220_v01.bsp",
    "spk/bc_mpo_mlt_50037_20270301_20290516_v01.bsp",
    "spk/bc_mpo_fcp_00237_20181020_20270411_v02.bsp",
]


def test_keeps_only_what_mio_mpo_positions_and_msasi_pointing_need_in_load_order():
    assert wanted_kernels(ESA_MK) == WANTED


def test_local_metakernel_loads_from_the_cache_folder(tmp_path):
    text = local_metakernel(["lsk/naif0012.tls"], tmp_path)
    assert f"PATH_VALUES = ( '{tmp_path}' )" in text
    assert "'$KERNELS/lsk/naif0012.tls'" in text
    assert text.startswith("\\begindata")


def _fake_fetch(urls):
    def fetch(url: str) -> bytes:
        urls.append(url)
        return ESA_MK.encode() if url.endswith("bc_plan.tm") else b"kernel"
    return fetch


def test_downloads_the_missing_kernels_once(tmp_path):
    urls = []
    metakernel = ensure_kernels(tmp_path, fetch=_fake_fetch(urls))
    assert Path(metakernel).read_text() == local_metakernel(WANTED, tmp_path)
    assert (tmp_path / "lsk/naif0012.tls").read_bytes() == b"kernel"
    assert len(urls) == 1 + len(WANTED)
    urls.clear()
    ensure_kernels(tmp_path, fetch=_fake_fetch(urls))
    assert urls == ["https://spiftp.esac.esa.int/data/SPICE/BEPICOLOMBO/kernels/mk/bc_plan.tm"]


def test_offline_falls_back_to_the_last_metakernel(tmp_path):
    first = ensure_kernels(tmp_path, fetch=_fake_fetch([]))

    def offline(url):
        raise OSError("no network")

    assert ensure_kernels(tmp_path, fetch=offline) == first
