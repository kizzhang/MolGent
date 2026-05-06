"""Structure preparation: PDBFixer + membrane embedding for ClC-family channels.

The ClC-2 TMD structures (7XJA, 8GQU) miss several loops and are deposited
without hydrogens; we run PDBFixer to repair, add hydrogens at pH 7.4, and
return an OpenMM Topology + positions ready for membrane embedding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from molgent.core.forcefield import make_forcefield

log = logging.getLogger(__name__)


@dataclass
class PreparedStructure:
    pdb_path: Path
    n_atoms: int
    n_residues: int
    n_chains: int


def fix_pdb(
    in_pdb: Path,
    out_pdb: Path,
    ph: float = 7.4,
    *,
    keep_heterogens: bool = False,
) -> PreparedStructure:
    """Add missing residues/atoms/hydrogens with PDBFixer."""

    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(in_pdb))
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    if not keep_heterogens:
        fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    with out_pdb.open("w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    n_atoms = sum(1 for _ in fixer.topology.atoms())
    n_residues = sum(1 for _ in fixer.topology.residues())
    n_chains = sum(1 for _ in fixer.topology.chains())
    return PreparedStructure(out_pdb, n_atoms, n_residues, n_chains)


def add_membrane_and_solvate(
    fixed_pdb: Path,
    out_pdb: Path,
    *,
    lipid: str = "POPC",
    padding_nm: float = 1.5,
    ionic_strength_M: float = 0.15,
    forcefield_files: tuple[str, ...] = (
        "amber14/protein.ff14SB.xml",
        "amber14/lipid17.xml",
        "amber14/tip3p.xml",
    ),
    ligand_sdf: Path | None = None,
    ligand_ff_xml: Path | None = None,
) -> PreparedStructure:
    """Embed the protein in a POPC bilayer and solvate with TIP3P + 150 mM NaCl.

    Uses ``openmm.app.Modeller.addMembrane`` so that the resulting box is
    ready for AMBER14SB + Lipid17 + TIP3P MD.
    """

    from openmm import unit
    from openmm.app import Modeller, PDBFile

    pdb = PDBFile(str(fixed_pdb))
    modeller = Modeller(pdb.topology, pdb.positions)
    forcefield = make_forcefield(
        forcefield_files,
        ligand_sdf=ligand_sdf,
        ligand_ff_xml=ligand_ff_xml,
    )
    modeller.addMembrane(
        forcefield,
        lipidType=lipid,
        membraneCenterZ=0 * unit.nanometer,
        minimumPadding=padding_nm * unit.nanometer,
        positiveIon="Na+",
        negativeIon="Cl-",
        ionicStrength=ionic_strength_M * unit.molar,
    )

    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    with out_pdb.open("w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)

    n_atoms = sum(1 for _ in modeller.topology.atoms())
    n_residues = sum(1 for _ in modeller.topology.residues())
    n_chains = sum(1 for _ in modeller.topology.chains())
    return PreparedStructure(out_pdb, n_atoms, n_residues, n_chains)
