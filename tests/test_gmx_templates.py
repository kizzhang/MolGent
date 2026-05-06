"""Sanity checks for the GROMACS pipeline scaffolding.

These tests don't shell out to ``gmx`` — they cover the parts that are pure
Python: MDP template rendering, gmx_MMPBSA decomposition parsing, and the
CHARMM-GUI bundle locator. Anything that needs a real ``gmx`` binary is left
for the manual ``cpu-verify`` run.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


def test_mdp_templates_render_with_expected_placeholders(tmp_path: Path) -> None:
    from molgent.core import gmx

    em = gmx._read_mdp_template("em.mdp")
    assert "integrator              = steep" in em
    assert "{" not in em  # em.mdp has no placeholders

    nvt = gmx._render_mdp(
        "nvt.mdp", tmp_path / "nvt.mdp",
        timestep_ps="0.0020", nsteps=50000,
        temperature_k="310.00", seed=42,
    )
    text = nvt.read_text()
    assert "tcoupl                  = v-rescale" in text
    assert "tc-grps                 = SOLU MEMB SOLV" in text
    assert "ref_t                   = 310.00  310.00  310.00" in text
    assert "{" not in text                          # all placeholders resolved

    npt = gmx._render_mdp(
        "npt.mdp", tmp_path / "npt.mdp",
        timestep_ps="0.0020", nsteps=500000,
        temperature_k="310.00", pressure_bar="1.000",
    )
    text = npt.read_text()
    assert "pcoupl                  = Parrinello-Rahman" in text
    assert "pcoupltype              = semiisotropic" in text
    assert "ref_p                   = 1.000 1.000" in text
    assert "{" not in text

    prod = gmx._render_mdp(
        "prod.mdp", tmp_path / "prod.mdp",
        timestep_ps="0.0020", nsteps=2_500_000,
        temperature_k="310.00", pressure_bar="1.000",
        nstenergy=500, nstlog=500, nstxout_compressed=5000,
    )
    text = prod.read_text()
    assert "nstxout-compressed      = 5000" in text
    # paper's "no position restraints in production"
    assert "-DPOSRES" not in text
    assert "{" not in text


def test_mdp_render_raises_for_missing_placeholder(tmp_path: Path) -> None:
    from molgent.core import gmx

    with pytest.raises(KeyError):
        gmx._render_mdp("nvt.mdp", tmp_path / "nvt.mdp",
                        timestep_ps="0.0020", nsteps=10)  # missing temperature_k, seed


def test_charmm_gui_bundle_locator(tmp_path: Path) -> None:
    from molgent.core import gmx

    # Lay out a fake CHARMM-GUI archive.
    base = tmp_path / "bound_tmd"
    inner = base / "charmm-gui-1234567890" / "gromacs"
    inner.mkdir(parents=True)
    (inner / "step5_input.gro").write_text("placeholder gro\n")
    (inner / "topol.top").write_text("placeholder top\n")

    found = gmx._find_gromacs_subdir(base)
    assert found == inner

    # Same content but flattened: gromacs/ directly under base
    base2 = tmp_path / "bound_tmd_flat"
    flat = base2 / "gromacs"
    flat.mkdir(parents=True)
    (flat / "step5_input.gro").write_text("ok\n")
    (flat / "topol.top").write_text("ok\n")
    assert gmx._find_gromacs_subdir(base2) == flat

    with pytest.raises(FileNotFoundError):
        gmx._find_gromacs_subdir(tmp_path / "nonexistent")


def test_decomp_parser_extracts_per_residue_total(tmp_path: Path) -> None:
    from molgent.core import mmpbsa_gmx

    # Hand-crafted snippet matching gmx_MMPBSA's per-residue table layout.
    decomp = textwrap.dedent("""\
        | GENERALIZED BORN

        Total Energy Decomposition:

        Residue          | Internal | van der Waals | Electrostatic | Polar Solvation | Non-Polar Solv. | TOTAL
        -------------------------------------------------------------------------------------------
        R LYS  204     |   0.000  0.000  0.000 |  -3.21  0.50  0.04 |  -0.45  0.20  0.02 |   1.34  0.30  0.03 |  -0.12  0.05  0.01 |  -2.44  0.60  0.04
        R SER  392     |   0.000  0.000  0.000 |  -1.10  0.30  0.02 |  -1.50  0.40  0.03 |   0.80  0.20  0.02 |  -0.10  0.04  0.01 |  -1.90  0.50  0.04
        R LYS 394:A   |   0.000  0.000  0.000 |  -2.10  0.40  0.03 |  -1.05  0.30  0.02 |   0.80  0.20  0.02 |  -0.05  0.02  0.01 |  -2.40  0.50  0.04

        Sidechain Energy Decomposition:
        R LYS  204     | 0.0 0.0 0.0 | -1.0 0.1 0.01 | -0.5 0.1 0.01 | 0.5 0.1 0.01 | -0.05 0.01 0.001 | -1.05 0.2 0.02
    """)
    path = tmp_path / "FINAL_DECOMP_MMPBSA.dat"
    path.write_text(decomp)

    rows = mmpbsa_gmx._parse_decomp_file(path)
    by_resid = {(r[0], r[1]): r for r in rows}

    # Three rows from the TOTAL block, none from the SIDECHAIN block.
    assert len(rows) == 3
    assert ("A", 204) in by_resid
    assert ("A", 392) in by_resid
    assert ("A", 394) in by_resid

    chain, idx, name, mean, std = by_resid[("A", 204)]
    assert name == "LYS"
    assert mean == pytest.approx(-2.44)
    assert std == pytest.approx(0.60)

    # Chain explicitly tagged as "A" via the "394:A" syntax should round-trip
    chain394, *_ = by_resid[("A", 394)]
    assert chain394 == "A"


def test_segid_translation_for_charmm_gui(tmp_path: Path) -> None:
    from molgent.workflows.clc2_ak42 import run_gromacs

    raw = [
        {"name": "K204_test", "chain": "A", "residue_index": 204,
         "residue_atom": "NZ", "ligand_resname": "GH6", "ligand_atom": "N08",
         "ligand_chain": "A"},
    ]
    out = run_gromacs._translate_distance_specs(raw)
    assert out[0].chain == "PROA"
    assert out[0].ligand_chain == "HETA"
