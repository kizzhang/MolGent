"""AK-42 (and arbitrary small-molecule) handling: SMILES/PDB -> SDF/MOL2/PDB.

Two source modes for the ligand 3D pose:

* ``smiles_to_sdf``: build a fresh ETKDG conformer from a SMILES string. Used
  when no experimental structure is available. The generated atom names are
  RDKit-style (C1, N1, ...).

* ``extract_from_pdb``: pull HETATM records of a CCD residue (e.g. ``GH6``)
  out of an experimental PDB, perceive bonds with RDKit using the SMILES as a
  template, and write SDF/MOL2/PDB while preserving the original CCD atom
  names (``N08``, ``O01``, ``O03`` ...). Use this when the cryo-EM bound pose
  is the desired starting point — it keeps the analysis distance specs in
  sync with the deposited PDB.

Force-field handoff:

* ``ff_xml``: optional pre-generated OpenMM ForceField XML for the AMBER/GAFF2
  reference pipeline (``molgent-clc2``).
* ``charmm_str``: optional CGenFF stream file for the GROMACS / CHARMM36m
  pipeline (``molgent-clc2-gmx``). Generated externally via cgenff.silcsbio.com
  or the licensed CGenFF binary; we just track the path here.
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
    ff_xml: Path | None = None       # pre-generated OpenMM ForceField file
    pdb_path: Path | None = None     # extracted-from-PDB ligand (CCD atom names)
    mol2_path: Path | None = None    # MOL2 export for CGenFF upload
    charmm_str: Path | None = None   # CGenFF .str (CHARMM36m parameters)


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


def _filter_pdb_block(pdb_path: Path, resname: str, chain: str) -> str:
    """Return a PDB text block containing only HETATM records for the residue.

    Keeps the original CCD atom names (cols 13-16) untouched and rewrites the
    chain to ``A`` so that downstream OpenBabel/RDKit parsers don't get
    confused by gaps. CONECT records are dropped — RDKit will rebuild bonds
    using the SMILES template.
    """

    keep_lines: list[str] = []
    with pdb_path.open() as f:
        for line in f:
            if not line.startswith(("HETATM", "ATOM  ")):
                continue
            if line[17:20].strip() != resname:
                continue
            if line[21:22] != chain:
                continue
            keep_lines.append(line)
    if not keep_lines:
        raise ValueError(
            f"No {resname} records on chain {chain!r} in {pdb_path}"
        )
    keep_lines.append("END\n")
    return "".join(keep_lines)


def extract_from_pdb(
    pdb_path: Path,
    resname: str,
    chain: str,
    smiles: str,
    out_dir: Path,
    *,
    name: str | None = None,
) -> Ligand:
    """Extract a CCD ligand from an experimental PDB and build SDF + MOL2.

    The 3D coordinates come from the PDB (cryo-EM / X-ray bound pose). Bond
    perception uses ``smiles`` as a template via
    ``RDKit.AllChem.AssignBondOrdersFromTemplate`` so that aromaticity and
    formal charges are correct. CCD atom names (e.g. ``N08``, ``O01``) are
    preserved on the output PDB; the SDF/MOL2 use the same atom labels in
    their atom-name fields where supported.

    Returns a :class:`Ligand` with ``sdf_path``, ``pdb_path``, ``mol2_path``
    set. ``charmm_str`` is left to the caller to populate after CGenFF.
    """

    from rdkit import Chem
    from rdkit.Chem import AllChem

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = name or resname

    pdb_block = _filter_pdb_block(pdb_path, resname, chain)
    raw = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
    if raw is None:
        raise RuntimeError(f"RDKit failed to parse extracted {resname} PDB block")

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"RDKit failed to parse template SMILES: {smiles!r}")

    # AssignBondOrdersFromTemplate refuses if explicit Hs are present; drop and
    # re-add later (CGenFF will assign its own H positions anyway).
    raw_no_h = Chem.RemoveHs(raw, sanitize=False)
    try:
        mol = AllChem.AssignBondOrdersFromTemplate(template, raw_no_h)
    except ValueError as exc:
        raise RuntimeError(
            f"Could not match SMILES template to extracted {resname} atoms "
            f"(check that the PDB residue has the expected heavy-atom count "
            f"and the SMILES is correct): {exc}"
        ) from exc

    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol, addCoords=True)
    mol.SetProp("_Name", name)

    pdb_out = out_dir / f"{name}_from_pdb.pdb"
    sdf_out = out_dir / f"{name}_from_pdb.sdf"
    mol2_out = out_dir / f"{name}_from_pdb.mol2"

    Chem.MolToPDBFile(mol, str(pdb_out), flavor=4)  # flavor=4: keep atom names

    sdf_writer = Chem.SDWriter(str(sdf_out))
    sdf_writer.write(mol)
    sdf_writer.close()

    # MOL2 export — RDKit ships an MOL2 writer only as part of OpenBabel; if
    # that is not available, fall back to writing a placeholder so the path
    # exists and CGenFF upload happens via SDF instead.
    try:
        from rdkit.Chem import rdmolfiles
        rdmolfiles.MolToMol2File(mol, str(mol2_out))
    except (AttributeError, RuntimeError):
        log.info("RDKit MOL2 writer unavailable; skipping %s", mol2_out.name)
        mol2_out = None  # type: ignore[assignment]

    log.info(
        "Extracted %s from %s chain %s: %d heavy atoms, %d bonds",
        resname, pdb_path.name, chain, mol.GetNumHeavyAtoms(), mol.GetNumBonds(),
    )

    return Ligand(
        name=name,
        smiles=smiles,
        sdf_path=sdf_out,
        pdb_path=pdb_out,
        mol2_path=mol2_out,
        charmm_str=None,
    )


def load_ligand(name: str, smiles: str, out_dir: Path) -> Ligand:
    """Build (or reuse) the SDF for a ligand and return a Ligand record."""

    out_dir = Path(out_dir)
    sdf_path = out_dir / f"{name}.sdf"
    if not sdf_path.exists():
        sdf_path = smiles_to_sdf(smiles, name, out_dir)
    ff_xml = out_dir / f"{name}.xml"
    charmm_str = out_dir / f"{name.lower()}.str"
    return Ligand(
        name=name,
        smiles=smiles,
        sdf_path=sdf_path,
        ff_xml=ff_xml if ff_xml.exists() else None,
        charmm_str=charmm_str if charmm_str.exists() else None,
    )


# AK-42 as deposited in RCSB CCD ligand GH6 for 8GQU.
AK42_SMILES = "OC(=O)c1cccnc1Nc2c(Cl)ccc(OCc3ccccc3)c2Cl"
