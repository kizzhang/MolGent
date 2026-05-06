"""Tests for :func:`molgent.core.ligand.extract_from_pdb`.

The function preserves CCD atom names from the deposited PDB so the analysis
distance specs (which reference ``N08``, ``O01``, ``O03``) keep working
against the cryo-EM bound pose.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from .conftest import HAS_RDKIT


# Tiny synthetic PDB containing two HETATM blocks: one for the residue we
# want (``GH6`` chain A, 26 heavy atoms — keeping it small with a fragment),
# plus a stray water that must be ignored.
SYNTH_PDB = textwrap.dedent("""\
HETATM    1  N08 GH6 A 901       0.000   0.000   0.000  1.00 20.00           N
HETATM    2  C01 GH6 A 901       1.330   0.000   0.000  1.00 20.00           C
HETATM    3  C02 GH6 A 901       2.000   1.220   0.000  1.00 20.00           C
HETATM    4  O01 GH6 A 901       1.300   2.300   0.000  1.00 20.00           O
HETATM    5  O03 GH6 A 901       3.300   1.200   0.000  1.00 20.00           O
HETATM    6  HXT GH6 A 901       3.700   2.080   0.000  1.00 20.00           H
HETATM    7  O   HOH B 100       8.000   8.000   8.000  1.00 30.00           O
END
""")

# Matches the heavy atoms above: N-C(=O)-OH with one ring bond closure
# (4 heavy atoms in skeleton + 2 explicit bonds); but RDKit's
# AssignBondOrdersFromTemplate is strict about heavy-atom counts so we use a
# minimal SMILES that exactly matches our 5 heavy atoms.
SYNTH_SMILES = "NCC(=O)O"  # glycine-like: 5 heavy atoms (N C C O O)


@pytest.mark.skipif(not HAS_RDKIT, reason="rdkit not installed")
def test_filter_pdb_block_keeps_only_target_residue(tmp_path: Path) -> None:
    from molgent.core.ligand import _filter_pdb_block

    pdb = tmp_path / "synth.pdb"
    pdb.write_text(SYNTH_PDB)
    block = _filter_pdb_block(pdb, resname="GH6", chain="A")
    # All 6 atoms of GH6 chain A, no HOH.
    assert block.count("GH6 A 901") == 6
    assert "HOH" not in block
    assert block.endswith("END\n")


@pytest.mark.skipif(not HAS_RDKIT, reason="rdkit not installed")
def test_filter_pdb_block_raises_for_missing_residue(tmp_path: Path) -> None:
    from molgent.core.ligand import _filter_pdb_block

    pdb = tmp_path / "synth.pdb"
    pdb.write_text(SYNTH_PDB)
    with pytest.raises(ValueError, match="No XYZ records"):
        _filter_pdb_block(pdb, resname="XYZ", chain="A")


@pytest.mark.skipif(not HAS_RDKIT, reason="rdkit not installed")
def test_extract_from_pdb_preserves_ccd_names(tmp_path: Path) -> None:
    """End-to-end: extract -> PDB output keeps N08/O01/O03 atom labels.

    The roundtrip would normally need a ligand whose heavy-atom count matches
    a SMILES template. Here we use a glycine-like 5-atom skeleton that maps
    cleanly to ``NCC(=O)O``. The point of this test is to lock in atom-name
    preservation, not to validate ChemDraw-grade chemistry.
    """

    from molgent.core import ligand

    pdb = tmp_path / "synth.pdb"
    pdb.write_text(SYNTH_PDB)

    out = tmp_path / "out"
    lig = ligand.extract_from_pdb(
        pdb_path=pdb, resname="GH6", chain="A",
        smiles=SYNTH_SMILES, out_dir=out, name="SYN",
    )
    assert lig.pdb_path is not None and lig.pdb_path.exists()
    text = lig.pdb_path.read_text()
    # CCD atom names from input survive RDKit roundtrip.
    for atomname in ("N08", "O01", "O03"):
        assert atomname in text, f"{atomname!r} missing from extracted PDB output"
    assert lig.sdf_path.exists()
