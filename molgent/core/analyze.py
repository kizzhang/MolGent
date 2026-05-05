"""Trajectory analyses tied to the paper's structural claims.

Three reproducible quantities:
  * RMSD of TMD backbone vs. apo reference, per replica.
  * RMSF per residue.
  * Time series of three diagnostic distances:
      - K204 NZ – AK-42 pyridine N
      - S392 OG – AK-42 carboxyl O
      - K394 backbone N – AK-42 carboxyl O
    AK-42 atom names come from the SDF / OpenMM-built residue.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class DistanceSpec:
    """One residue.atom <-> ligand.atom distance to track."""

    name: str
    residue_index: int     # 1-based, matches PDB numbering
    residue_atom: str      # e.g. "NZ"
    ligand_resname: str    # e.g. "AK4" or "LIG"
    ligand_atom: str       # e.g. "N7"


@dataclass
class AnalysisConfig:
    topology_pdb: Path
    trajectory_dcd: Path
    out_dir: Path
    reference_pdb: Path | None = None
    distances: list[DistanceSpec] = field(default_factory=list)
    occupancy_cutoff_nm: float = 0.35   # 3.5 Å H-bond cutoff


@dataclass
class AnalysisResult:
    rmsd_csv: Path
    rmsf_csv: Path
    distances_csv: Path | None
    occupancy_json: Path | None


def _select_one(universe, sel: str):
    sub = universe.select_atoms(sel)
    if sub.n_atoms != 1:
        raise ValueError(f"Selection {sel!r} matched {sub.n_atoms} atoms (want 1)")
    return sub[0]


def run_analysis(cfg: AnalysisConfig) -> AnalysisResult:
    import MDAnalysis as mda
    from MDAnalysis.analysis import rms

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    u = mda.Universe(str(cfg.topology_pdb), str(cfg.trajectory_dcd))

    # RMSD vs reference (or first frame)
    if cfg.reference_pdb is not None:
        ref = mda.Universe(str(cfg.reference_pdb))
    else:
        ref = mda.Universe(str(cfg.topology_pdb))
    R = rms.RMSD(u, ref, select="protein and name CA")
    R.run()
    rmsd_csv = cfg.out_dir / "rmsd.csv"
    np.savetxt(
        rmsd_csv, R.results.rmsd, delimiter=",",
        header="frame,time_ps,rmsd_A", comments="",
    )

    # RMSF
    protein = u.select_atoms("protein and name CA")
    rmsf = rms.RMSF(protein).run()
    rmsf_csv = cfg.out_dir / "rmsf.csv"
    with rmsf_csv.open("w") as f:
        f.write("residue_index,residue_name,rmsf_A\n")
        for atom, val in zip(protein, rmsf.results.rmsf):
            f.write(f"{atom.resid},{atom.resname},{val:.3f}\n")

    distances_csv = None
    occupancy_json = None
    if cfg.distances:
        # Pre-resolve atom indices
        pairs = []
        for spec in cfg.distances:
            res_atom = _select_one(
                u, f"resid {spec.residue_index} and name {spec.residue_atom}"
            )
            lig_atom = _select_one(
                u, f"resname {spec.ligand_resname} and name {spec.ligand_atom}"
            )
            pairs.append((spec.name, res_atom.index, lig_atom.index))

        n_frames = len(u.trajectory)
        n_pairs = len(pairs)
        data = np.zeros((n_frames, n_pairs + 1))
        for fi, ts in enumerate(u.trajectory):
            data[fi, 0] = ts.time
            for pi, (_, ai, bi) in enumerate(pairs):
                data[fi, pi + 1] = (
                    np.linalg.norm(u.atoms[ai].position - u.atoms[bi].position) / 10.0
                )  # Å -> nm

        distances_csv = cfg.out_dir / "distances.csv"
        header = "time_ps," + ",".join(p[0] for p in pairs)
        np.savetxt(distances_csv, data, delimiter=",", header=header, comments="")

        occupancy = {
            name: float(np.mean(data[:, i + 1] <= cfg.occupancy_cutoff_nm))
            for i, (name, *_rest) in enumerate(pairs)
        }
        occupancy_json = cfg.out_dir / "hbond_occupancy.json"
        occupancy_json.write_text(json.dumps(occupancy, indent=2))

    return AnalysisResult(rmsd_csv, rmsf_csv, distances_csv, occupancy_json)
