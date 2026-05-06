"""Shared OpenMM ForceField construction helpers."""

from __future__ import annotations

from pathlib import Path


def make_forcefield(
    forcefield_files: tuple[str, ...],
    *,
    ligand_sdf: Path | None = None,
    ligand_ff_xml: Path | None = None,
):
    """Create an OpenMM ForceField, optionally registering a small molecule."""

    from openmm.app import ForceField

    forcefield = ForceField(*forcefield_files)

    if ligand_ff_xml is not None and Path(ligand_ff_xml).exists():
        forcefield.loadFile(str(ligand_ff_xml))
        return forcefield

    if ligand_sdf is not None:
        try:
            from openff.toolkit.topology import Molecule
            from openmmforcefields.generators import GAFFTemplateGenerator
        except ImportError as exc:
            raise ImportError(
                "Ligand SDF supplied but openff-toolkit / openmmforcefields are "
                "not importable. Install them from conda-forge together with "
                "AmberTools, or pre-generate an OpenMM ForceField XML and pass "
                "it as ligand_ff_xml."
            ) from exc

        molecule = Molecule.from_file(str(ligand_sdf))
        gaff = GAFFTemplateGenerator(molecules=molecule, forcefield="gaff-2.11")
        forcefield.registerTemplateGenerator(gaff.generator)

    return forcefield
