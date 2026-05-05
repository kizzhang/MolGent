"""Tiny end-to-end smoke test for the OpenMM stack — alanine dipeptide.

Skipped unless openmm + pdbfixer are importable. Designed to run on the
sandbox's CPU in well under a minute, so CI catches obvious wiring bugs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import HAS_OPENMM, HAS_PDBFIXER

ALA_DIPEPTIDE_PDB = """\
REMARK  alanine dipeptide (ACE-ALA-NME)
ATOM      1  HH31 ACE A   1       2.000   1.000   1.000  1.00  0.00           H
ATOM      2  CH3  ACE A   1       2.000   2.000   1.000  1.00  0.00           C
ATOM      3  C    ACE A   1       3.500   2.000   1.000  1.00  0.00           C
ATOM      4  O    ACE A   1       4.000   3.100   1.000  1.00  0.00           O
ATOM      5  N    ALA A   2       4.200   0.900   1.000  1.00  0.00           N
ATOM      6  CA   ALA A   2       5.700   0.900   1.000  1.00  0.00           C
ATOM      7  CB   ALA A   2       6.200  -0.500   1.000  1.00  0.00           C
ATOM      8  C    ALA A   2       6.300   1.700   2.200  1.00  0.00           C
ATOM      9  O    ALA A   2       6.000   2.900   2.300  1.00  0.00           O
ATOM     10  N    NME A   3       7.200   1.000   3.000  1.00  0.00           N
ATOM     11  CH3  NME A   3       7.900   1.700   4.100  1.00  0.00           C
END
"""


@pytest.mark.skipif(not (HAS_OPENMM and HAS_PDBFIXER),
                    reason="openmm+pdbfixer required for the MD smoke test")
def test_minimize_short(tmp_path: Path) -> None:
    from openmm import LangevinMiddleIntegrator, Platform, unit
    from openmm.app import (ForceField, HBonds, NoCutoff, PDBFile, Simulation)
    from pdbfixer import PDBFixer

    raw = tmp_path / "ala.pdb"
    raw.write_text(ALA_DIPEPTIDE_PDB)

    fixer = PDBFixer(filename=str(raw))
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)
    fixed = tmp_path / "ala_fixed.pdb"
    with fixed.open("w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

    pdb = PDBFile(str(fixed))
    ff = ForceField("amber14-all.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff,
                             constraints=HBonds)
    integ = LangevinMiddleIntegrator(310 * unit.kelvin, 1.0 / unit.picosecond,
                                     1.0 * unit.femtosecond)
    sim = Simulation(pdb.topology, system, integ, Platform.getPlatformByName("Reference"))
    sim.context.setPositions(pdb.positions)
    sim.minimizeEnergy(maxIterations=50)
    sim.step(100)

    e = sim.context.getState(getEnergy=True).getPotentialEnergy()
    assert e.value_in_unit(unit.kilojoule_per_mole) < 1e6
