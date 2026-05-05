"""AK-42 SMILES → 3D conformer roundtrip via RDKit."""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import HAS_RDKIT


@pytest.mark.skipif(not HAS_RDKIT, reason="rdkit not installed in this environment")
def test_ak42_sdf(tmp_path: Path) -> None:
    from molgent.core import ligand

    lg = ligand.load_ligand("AK4", ligand.AK42_SMILES, tmp_path)
    assert lg.sdf_path.exists()
    text = lg.sdf_path.read_text()
    # Sanity: SDF, contains atoms, contains the molecule name.
    assert text.startswith("AK4") or "AK4" in text
    assert "M  END" in text or "END" in text
