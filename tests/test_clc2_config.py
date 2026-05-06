"""Regression tests for the ClC-2 / AK-42 reproduction config.

The original config swapped GH6's two carboxyl oxygens (O01 vs O03) in two
distance specs, which caused K394(N)–O distance to drift far from the
paper's value and gave 0% H-bond occupancy. These tests pin the corrected
mapping so the bug cannot return silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import HAS_YAML

CONFIG = Path(__file__).resolve().parents[1] / "molgent/workflows/clc2_ak42/config.yaml"


@pytest.mark.skipif(not HAS_YAML, reason="pyyaml not installed")
def test_distance_specs_use_correct_carboxyl_oxygens() -> None:
    import yaml

    cfg = yaml.safe_load(CONFIG.read_text())
    by_name = {d["name"]: d for d in cfg["distances"]}

    s392 = by_name["S392(OG)-AK42(COOH)"]
    k394 = by_name["K394(N)-AK42(COOH)"]
    k204 = by_name["K204(NZ)-AK42(pyN)"]

    # Locked by 8GQU geometry: S392(OG) is 3.51 Å from O01, 3.70 Å from O03.
    assert s392["ligand_atom"] == "O01", (
        "S392(OG) should pair with O01 — see 8GQU chain A geometry"
    )
    # K394 backbone N is 2.61 Å from O03, 4.55 Å from O01.
    assert k394["ligand_atom"] == "O03", (
        "K394(N) should pair with O03 — closest carboxyl O at 2.61 Å"
    )
    # K204(NZ) lines up with the pyridine nitrogen N08 (no swap risk).
    assert k204["ligand_atom"] == "N08"


@pytest.mark.skipif(not HAS_YAML, reason="pyyaml not installed")
def test_gromacs_section_matches_paper_protocol() -> None:
    import yaml

    cfg = yaml.safe_load(CONFIG.read_text())
    g = cfg["gromacs"]
    assert g["thermostat"] == "v-rescale"
    assert g["barostat"] == "Parrinello-Rahman"
    assert g["pcoupltype"] == "semiisotropic"
    assert g["temperature_K"] == pytest.approx(310.0)
    assert g["nvt_ps"] == pytest.approx(100.0)        # paper
    assert g["production_ns"] == pytest.approx(5.0)   # paper
    assert g["n_replicas"] == 5                       # paper
    # CHARMM36m force-switched LJ (rvdw 1.0–1.2 nm)
    assert g["rvdw_nm"] == pytest.approx(1.2)
    assert g["rvdw_switch_nm"] == pytest.approx(1.0)
    assert g["rcoulomb_nm"] == pytest.approx(1.2)


@pytest.mark.skipif(not HAS_YAML, reason="pyyaml not installed")
def test_expected_yaml_has_crystal_baselines() -> None:
    import yaml

    expected_path = CONFIG.with_name("expected.yaml")
    expected = yaml.safe_load(expected_path.read_text())
    hb = expected["hbond_distances_A"]
    # The crystal_8gqu values must match what we measured on chain A — the
    # comment in expected.yaml says these reflect raw deposited coordinates.
    assert hb["K204_NZ_to_pyridine_N"]["crystal_8gqu"] == pytest.approx(3.68, abs=0.05)
    assert hb["S392_OG_to_carboxyl_O"]["crystal_8gqu"] == pytest.approx(3.51, abs=0.05)
    assert hb["K394_N_to_carboxyl_O"]["crystal_8gqu"] == pytest.approx(2.61, abs=0.05)
