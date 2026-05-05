"""OpenMM MD pipeline: build_system -> minimize -> NVT -> NPT -> production.

Designed for membrane systems with an optional small-molecule ligand.
The CUDA platform is preferred when present; otherwise falls back to CPU
(for sandbox smoke tests only — production runs need a GPU).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


@dataclass
class MDConfig:
    out_dir: Path
    name: str = "system"
    temperature_K: float = 310.0
    pressure_bar: float = 1.0
    timestep_fs: float = 4.0
    nonbonded_cutoff_nm: float = 1.0
    hmr_amu: float = 1.5
    minimize_max_iter: int = 5000
    nvt_steps: int = 125_000          # 500 ps at 4 fs
    npt_steps: int = 250_000          # 1 ns
    production_ns: float = 100.0
    write_interval_ps: float = 100.0
    seed: int = 1
    forcefield_files: tuple[str, ...] = (
        "amber14-all.xml",
        "amber14/lipid17.xml",
        "amber14/tip3p.xml",
    )
    ligand_sdf: Path | None = None    # if set, GAFFTemplateGenerator wires it in
    ligand_ff_xml: Path | None = None # alternative: pre-generated OpenMM FF


@dataclass
class MDResult:
    name: str
    topology_path: Path
    trajectory_path: Path
    state_path: Path
    log_path: Path
    walltime_s: float
    n_atoms: int


def _platform():
    from openmm import Platform

    for name in ("CUDA", "OpenCL", "CPU", "Reference"):
        try:
            return Platform.getPlatformByName(name)
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("No OpenMM platform available")


def _make_forcefield(cfg: MDConfig):
    from openmm.app import ForceField

    forcefield = ForceField(*cfg.forcefield_files)

    if cfg.ligand_ff_xml is not None and Path(cfg.ligand_ff_xml).exists():
        forcefield.loadFile(str(cfg.ligand_ff_xml))
        return forcefield

    if cfg.ligand_sdf is not None:
        try:
            from openff.toolkit.topology import Molecule
            from openmmforcefields.generators import GAFFTemplateGenerator
        except ImportError as exc:
            raise ImportError(
                "Ligand SDF supplied but openff-toolkit / openmmforcefields are "
                "not importable. Install via conda on the GPU box, or pre-generate "
                "an OpenMM ForceField XML and pass it as ligand_ff_xml."
            ) from exc
        mol = Molecule.from_file(str(cfg.ligand_sdf))
        gaff = GAFFTemplateGenerator(molecules=mol, forcefield="gaff-2.11")
        forcefield.registerTemplateGenerator(gaff.generator)

    return forcefield


def build_system(prepared_pdb: Path, cfg: MDConfig):
    """Return (topology, positions, system, integrator) ready to simulate."""

    from openmm import LangevinMiddleIntegrator, MonteCarloMembraneBarostat, unit
    from openmm.app import HBonds, PDBFile, PME

    pdb = PDBFile(str(prepared_pdb))
    forcefield = _make_forcefield(cfg)

    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=cfg.nonbonded_cutoff_nm * unit.nanometer,
        constraints=HBonds,
        rigidWater=True,
        hydrogenMass=cfg.hmr_amu * unit.amu,
    )

    barostat = MonteCarloMembraneBarostat(
        cfg.pressure_bar * unit.bar,
        0 * unit.bar * unit.nanometer,  # surface tension
        cfg.temperature_K * unit.kelvin,
        MonteCarloMembraneBarostat.XYIsotropic,
        MonteCarloMembraneBarostat.ZFree,
    )
    system.addForce(barostat)

    integrator = LangevinMiddleIntegrator(
        cfg.temperature_K * unit.kelvin,
        1.0 / unit.picosecond,
        cfg.timestep_fs * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(cfg.seed)
    return pdb.topology, pdb.positions, system, integrator


def run_md(prepared_pdb: Path, cfg: MDConfig) -> MDResult:
    """Full pipeline: minimize, NVT/NPT equilibration, production."""

    from openmm import unit
    from openmm.app import DCDReporter, PDBFile, Simulation, StateDataReporter

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    topology, positions, system, integrator = build_system(prepared_pdb, cfg)

    sim = Simulation(topology, system, integrator, _platform())
    sim.context.setPositions(positions)

    log.info("Minimizing %d atoms", system.getNumParticles())
    sim.minimizeEnergy(maxIterations=cfg.minimize_max_iter)

    sim.context.setVelocitiesToTemperature(cfg.temperature_K * unit.kelvin, cfg.seed)

    state_data_path = cfg.out_dir / f"{cfg.name}.log"
    state_reporter = StateDataReporter(
        str(state_data_path),
        1000,
        step=True, potentialEnergy=True, temperature=True,
        volume=True, density=True, speed=True,
    )
    sim.reporters.append(state_reporter)

    log.info("NVT equilibration: %d steps", cfg.nvt_steps)
    sim.step(cfg.nvt_steps)
    log.info("NPT equilibration: %d steps", cfg.npt_steps)
    sim.step(cfg.npt_steps)

    write_every_steps = max(1, int(round(cfg.write_interval_ps * 1000.0 / cfg.timestep_fs)))
    traj_path = cfg.out_dir / f"{cfg.name}.dcd"
    sim.reporters.append(DCDReporter(str(traj_path), write_every_steps))

    production_steps = int(round(cfg.production_ns * 1e6 / cfg.timestep_fs))
    log.info("Production: %.1f ns (%d steps)", cfg.production_ns, production_steps)

    t0 = time.time()
    sim.step(production_steps)
    walltime = time.time() - t0

    state = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    state_path = cfg.out_dir / f"{cfg.name}_final.xml"
    with state_path.open("w") as f:
        from openmm import XmlSerializer

        f.write(XmlSerializer.serialize(state))

    topo_path = cfg.out_dir / f"{cfg.name}_topology.pdb"
    with topo_path.open("w") as f:
        PDBFile.writeFile(topology, state.getPositions(), f, keepIds=True)

    return MDResult(
        name=cfg.name,
        topology_path=topo_path,
        trajectory_path=traj_path,
        state_path=state_path,
        log_path=state_data_path,
        walltime_s=walltime,
        n_atoms=system.getNumParticles(),
    )


def run_replicas(prepared_pdb: Path, base_cfg: MDConfig, n_replicas: int) -> list[MDResult]:
    results = []
    for i in range(n_replicas):
        rep_cfg = MDConfig(
            **{**base_cfg.__dict__, "out_dir": base_cfg.out_dir / f"rep{i}",
               "name": f"{base_cfg.name}_rep{i}", "seed": base_cfg.seed + i}
        )
        results.append(run_md(prepared_pdb, rep_cfg))
    return results
