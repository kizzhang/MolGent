"""Per-residue MM/GBSA decomposition using OpenMM.

This is a deliberately small implementation that mirrors the *interpretation*
of MM/PBSA reported in Ma et al. 2023 — per-residue contributions to ligand
binding affinity — without depending on AmberTools / gmx_MMPBSA.

Procedure per saved trajectory frame:
  1. Strip waters/ions/lipids from the snapshot, retain protein + ligand.
  2. Re-build an implicit-solvent OpenMM ``System`` (GBn2) for:
       a) complex
       b) protein only
       c) ligand only
     Energy of binding ΔE_bind(frame) = E_complex - E_protein - E_ligand.
  3. Decompose by residue: temporarily zero non-bonded parameters of every
     other residue (and the ligand re-enabled / disabled) to obtain the
     pairwise residue↔ligand interaction energy.
  4. Average over frames; report top contributors.

The output is a CSV ``per_residue.csv`` with columns
``residue_index, residue_name, chain, dG_kcal_mol, std`` and a JSON summary
of the top-K residues with their values.
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molgent.core.forcefield import make_forcefield

log = logging.getLogger(__name__)

KJ_PER_KCAL = 4.184


@dataclass
class MMPBSAConfig:
    topology_pdb: Path
    trajectory_dcd: Path
    ligand_resname: str
    out_dir: Path
    forcefield_files: tuple[str, ...] = (
        "amber14/protein.ff14SB.xml",
        "implicit/gbn2.xml",
    )
    ligand_sdf: Path | None = None
    ligand_ff_xml: Path | None = None
    stride: int = 10        # use every Nth frame
    max_frames: int | None = None
    temperature_K: float = 310.0


@dataclass
class MMPBSAResult:
    csv_path: Path
    json_path: Path
    top_residues: list[tuple[str, float]]
    mean_dG_total_kcal: float


def _iter_frames(top_pdb: Path, traj_dcd: Path, stride: int, max_frames: int | None):
    import mdtraj as md

    t = md.load(str(traj_dcd), top=str(top_pdb), stride=stride)
    n = t.n_frames if max_frames is None else min(t.n_frames, max_frames)
    for i in range(n):
        yield t[i]


def _build_implicit_system(pdb_path: Path, cfg: MMPBSAConfig):
    from openmm.app import NoCutoff, PDBFile

    ff = make_forcefield(
        cfg.forcefield_files,
        ligand_sdf=cfg.ligand_sdf,
        ligand_ff_xml=cfg.ligand_ff_xml,
    )

    pdb = PDBFile(str(pdb_path))
    system = ff.createSystem(
        pdb.topology,
        nonbondedMethod=NoCutoff,
        constraints=None,
        rigidWater=True,
    )
    return pdb.topology, pdb.positions, system


def _energy(topology, positions, system) -> float:
    """Single-point potential energy in kJ/mol on the Reference platform."""

    from openmm import LangevinMiddleIntegrator, Platform, unit
    from openmm.app import Simulation

    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond
    )
    platform = Platform.getPlatformByName("Reference")
    sim = Simulation(topology, system, integrator, platform)
    sim.context.setPositions(positions)
    state = sim.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def _frame_to_pdbs(frame, ligand_resname: str, work_dir: Path) -> tuple[Path, Path, Path]:
    """Write complex / protein-only / ligand-only PDBs for a single frame."""


    work_dir.mkdir(parents=True, exist_ok=True)
    keep = frame.topology.select(f"protein or resname {ligand_resname}")
    sub = frame.atom_slice(keep)

    cpx = work_dir / "complex.pdb"
    sub.save_pdb(str(cpx))

    prot_idx = sub.topology.select("protein")
    lig_idx = sub.topology.select(f"resname {ligand_resname}")
    if len(lig_idx) == 0:
        raise RuntimeError(f"Ligand resname {ligand_resname!r} not found in frame")

    prot = sub.atom_slice(prot_idx)
    lig = sub.atom_slice(lig_idx)
    prot_pdb = work_dir / "protein.pdb"
    lig_pdb = work_dir / "ligand.pdb"
    prot.save_pdb(str(prot_pdb))
    lig.save_pdb(str(lig_pdb))
    return cpx, prot_pdb, lig_pdb


def _per_residue_interaction(complex_pdb: Path, ligand_resname: str, cfg: MMPBSAConfig
                             ) -> dict[tuple[str, int, str], float]:
    """For one frame: residue-by-residue ligand interaction energy in kJ/mol.

    Implementation: load complex with implicit-solvent FF; iterate residues;
    for each residue, freeze coordinates and compute the difference between
    full energy and the energy with that residue's nonbonded charges/sigmas
    zeroed. The signed difference is attributed as that residue's
    contribution. This is an approximation valid for short-range pairwise
    decompositions and is what we need to rank top contributors.
    """

    from openmm import NonbondedForce

    topology, positions, system = _build_implicit_system(complex_pdb, cfg)

    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    # Snapshot of original parameters
    n = nb.getNumParticles()
    orig = [nb.getParticleParameters(i) for i in range(n)]

    # Identify ligand particles (we won't zero them; we zero each residue)
    res_atoms: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for atom in topology.atoms():
        res = atom.residue
        try:
            resid = int(res.id)
        except (TypeError, ValueError):
            resid = res.index
        key = (res.chain.id, resid, res.name)
        res_atoms[key].append(atom.index)

    # Baseline energy
    e_full = _energy(topology, positions, system)

    out: dict[tuple[str, int, str], float] = {}
    ligand_keys = [k for k in res_atoms if k[2] == ligand_resname]
    if not ligand_keys:
        raise RuntimeError(f"No residue named {ligand_resname} in topology")

    for key, atom_ids in res_atoms.items():
        if key in ligand_keys:
            continue
        # Zero the residue's charges + LJ
        for i in atom_ids:
            nb.setParticleParameters(i, 0.0, 0.0, 0.0)
        # The new Simulation picks up the modified parameters when its Context
        # is created, so a single-point energy here reflects the perturbation.
        e_off = _energy(topology, positions, system)
        out[key] = e_full - e_off
        # Restore parameters
        for i, params in zip(atom_ids, [orig[a] for a in atom_ids]):
            nb.setParticleParameters(i, *params)

    return out


def run_mmpbsa(cfg: MMPBSAConfig, top_k: int = 10) -> MMPBSAResult:
    """Iterate frames, accumulate per-residue contributions, write outputs."""

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    accum: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    totals: list[float] = []

    work = cfg.out_dir / "frames"
    work.mkdir(exist_ok=True)

    for fi, frame in enumerate(_iter_frames(cfg.topology_pdb, cfg.trajectory_dcd,
                                            cfg.stride, cfg.max_frames)):
        cpx_pdb, prot_pdb, lig_pdb = _frame_to_pdbs(frame, cfg.ligand_resname,
                                                    work / f"f{fi:05d}")
        topo_c, pos_c, sys_c = _build_implicit_system(cpx_pdb, cfg)
        topo_p, pos_p, sys_p = _build_implicit_system(prot_pdb, cfg)
        topo_l, pos_l, sys_l = _build_implicit_system(lig_pdb, cfg)
        e_c = _energy(topo_c, pos_c, sys_c)
        e_p = _energy(topo_p, pos_p, sys_p)
        e_l = _energy(topo_l, pos_l, sys_l)
        totals.append((e_c - e_p - e_l) / KJ_PER_KCAL)

        per_res = _per_residue_interaction(cpx_pdb, cfg.ligand_resname, cfg)
        for k, v in per_res.items():
            accum[k].append(v / KJ_PER_KCAL)

        log.info("frame %d: ΔE_bind=%.2f kcal/mol (n_res=%d)", fi, totals[-1], len(per_res))

    csv_path = cfg.out_dir / "per_residue.csv"
    json_path = cfg.out_dir / "summary.json"

    rows = []
    for (chain, idx, name), values in accum.items():
        arr = np.asarray(values)
        rows.append((chain, idx, name, float(arr.mean()), float(arr.std(ddof=0))))
    rows.sort(key=lambda r: abs(r[3]), reverse=True)

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chain", "residue_index", "residue_name", "dG_kcal_mol", "std_kcal_mol"])
        w.writerows(rows)

    top = [(f"{name}{idx}", dg) for chain, idx, name, dg, _ in rows[:top_k]]
    summary = {
        "mean_dG_total_kcal": float(np.mean(totals)),
        "std_dG_total_kcal": float(np.std(totals)),
        "n_frames": len(totals),
        "top_residues": top,
    }
    json_path.write_text(json.dumps(summary, indent=2))

    return MMPBSAResult(
        csv_path=csv_path,
        json_path=json_path,
        top_residues=top,
        mean_dG_total_kcal=summary["mean_dG_total_kcal"],
    )
