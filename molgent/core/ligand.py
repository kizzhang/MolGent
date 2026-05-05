"""AK-42 (and arbitrary small-molecule) handling: SMILES -> 3D conformer -> SDF.

The ligand force-field path is two-track:
  * The default writes a 3D SDF; the MD module then asks
    ``openmmforcefields.GAFFTemplateGenerator`` to generate GAFF2 parameters
    on the fly. That dependency is only available with conda (it pulls in
    openff-toolkit + ambertools), so it's gated to the user's GPU box.
  * For pip-only (sandbox) workflows, you may pre-generate an OpenMM
    ForceField XML for the ligand once on the GPU box and commit it to
    ``data/ligands/<name>.xml``; ``md.build_system`` will pick that up.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Ligand:
    name: str
    smiles: str
    sdf_path: Path
    ff_xml: Path | None = None  # optional pre-generated OpenMM ForceField file


def smiles_to_sdf(smiles: str, name: str, out_dir: Path, seed: int = 42) -> Path:
    """Build a 3D conformer from SMILES, MMFF94-minimize, write SDF."""

    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        # fallback: try with random coordinates allowed
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise RuntimeError(f"ETKDGv3 embedding failed for {name}")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    mol.SetProp("_Name", name)

    out_dir.mkdir(parents=True, exist_ok=True)
    sdf_path = out_dir / f"{name}.sdf"
    writer = Chem.SDWriter(str(sdf_path))
    writer.write(mol)
    writer.close()
    return sdf_path


def load_ligand(name: str, smiles: str, out_dir: Path) -> Ligand:
    """Build (or reuse) the SDF for a ligand and return a Ligand record."""

    out_dir = Path(out_dir)
    sdf_path = out_dir / f"{name}.sdf"
    if not sdf_path.exists():
        sdf_path = smiles_to_sdf(smiles, name, out_dir)
    ff_xml = out_dir / f"{name}.xml"
    return Ligand(
        name=name,
        smiles=smiles,
        sdf_path=sdf_path,
        ff_xml=ff_xml if ff_xml.exists() else None,
    )


# AK-42, the ClC-2-selective inhibitor characterised by Koster et al. (PNAS 2020,
# doi:10.1073/pnas.2009977117). SMILES taken from the original disclosure.
AK42_SMILES = "OC(=O)c1cc(F)cc(c1)Oc2ccc3nc(NC(=O)c4cccnc4)sc3c2"
