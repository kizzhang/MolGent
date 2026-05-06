"""Direct (non-agent) entrypoint that reproduces Ma et al. 2023.

Usage:
    molgent-clc2 run --config molgent/workflows/clc2_ak42/config.yaml

The agent CLI (``molgent``) drives the same steps but lets the model decide
ordering and recovery; this script is the deterministic baseline used in CI
and as a fallback when no API key is available.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from molgent.core.forcefield import make_forcefield
from molgent.core import analyze, fetch, figures, ligand, md, mmpbsa, prepare

log = logging.getLogger("clc2_ak42")
REPO_ROOT = Path(__file__).resolve().parents[3]


def _abspath(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def _stage(state: str, pdb_id: str, paths: dict, lig_sdf: Path | None) -> dict:
    """Fetch + fix + (apo-only) embed for one state. Bound state expects a
    membrane-embedded system that already includes the ligand; for the
    deposited 8GQU (TMD with bound AK-42) we pass the ligand SDF when
    embedding so the OpenMM ForceField can parameterise it via GAFF."""

    out = _abspath(paths["out_dir"]) / state
    out.mkdir(parents=True, exist_ok=True)

    fetched = fetch.fetch_pdb(pdb_id, _abspath(paths["structure_cache"]))
    fixed = prepare.fix_pdb(
        fetched.path,
        out / f"{pdb_id}_fixed.pdb",
        keep_heterogens=lig_sdf is not None,
    )
    embedded = prepare.add_membrane_and_solvate(
        fixed.pdb_path,
        out / f"{pdb_id}_membrane.pdb",
        ligand_sdf=lig_sdf,
    )
    return {"state": state, "pdb_id": pdb_id, "fixed": str(fixed.pdb_path),
            "embedded": str(embedded.pdb_path),
            "ligand_sdf": str(lig_sdf) if lig_sdf else None}


def _run_state_md(stage: dict, lig: ligand.Ligand | None, md_cfg: dict, out_root: Path) -> list[md.MDResult]:
    base = md.MDConfig(
        out_dir=out_root / stage["state"] / "md",
        name=stage["state"],
        temperature_K=md_cfg["temperature_K"],
        pressure_bar=md_cfg["pressure_bar"],
        timestep_fs=md_cfg["timestep_fs"],
        hmr_amu=md_cfg["hmr_amu"],
        nonbonded_cutoff_nm=md_cfg["nonbonded_cutoff_nm"],
        nvt_steps=md_cfg["nvt_steps"],
        npt_steps=md_cfg["npt_steps"],
        production_ns=md_cfg["production_ns"],
        write_interval_ps=md_cfg["write_interval_ps"],
        ligand_sdf=lig.sdf_path if lig else None,
        ligand_ff_xml=lig.ff_xml if (lig and lig.ff_xml) else None,
    )
    return md.run_replicas(Path(stage["embedded"]), base, md_cfg["n_replicas"])


def reproduce(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    paths = cfg["paths"]
    out_root = _abspath(paths["out_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    log.info("Step 1/6 — prepare AK-42")
    lig = ligand.load_ligand(cfg["ligand"]["name"], cfg["ligand"]["smiles"],
                             _abspath(paths["ligand_dir"]))

    log.info("Step 2/6 — stage structures (apo TMD + AK-42-bound TMD)")
    apo = _stage("apo_tmd", cfg["paper"]["pdb_apo_tmd"], paths, None)
    bound = _stage("bound_tmd", cfg["paper"]["pdb_bound_tmd"], paths, lig.sdf_path)

    log.info("Step 3/6 — MD")
    _run_state_md(apo, None, cfg["md"], out_root)
    bound_runs = _run_state_md(bound, lig, cfg["md"], out_root)

    log.info("Step 4/6 — MM/GBSA on the bound trajectories")
    mmpbsa_summary = None
    for r in bound_runs:
        mres = mmpbsa.run_mmpbsa(mmpbsa.MMPBSAConfig(
            topology_pdb=r.topology_path, trajectory_dcd=r.trajectory_path,
            ligand_resname=cfg["mmpbsa"]["ligand_resname"],
            out_dir=r.trajectory_path.parent / "mmpbsa",
            ligand_sdf=lig.sdf_path,
            ligand_ff_xml=lig.ff_xml,
            stride=cfg["mmpbsa"]["stride"],
            max_frames=cfg["mmpbsa"]["max_frames"],
        ))
        if mmpbsa_summary is None:
            mmpbsa_summary = mres

    log.info("Step 5/6 — trajectory analysis")
    analysis_results = []
    for r in bound_runs:
        a = analyze.run_analysis(analyze.AnalysisConfig(
            topology_pdb=r.topology_path, trajectory_dcd=r.trajectory_path,
            out_dir=r.trajectory_path.parent / "analysis",
            distances=[analyze.DistanceSpec(**d) for d in cfg["distances"]],
        ))
        analysis_results.append(a)

    log.info("Step 6/6 — figures + report")
    fig_dir = out_root / "figures"
    figs = {
        "per_residue_dG": figures.plot_per_residue_dG(
            mmpbsa_summary.csv_path, fig_dir / "per_residue_dG.png"),
        "distances": figures.plot_distances(
            analysis_results[0].distances_csv, fig_dir / "distances.png"),
        "rmsf": figures.plot_rmsf(
            analysis_results[0].rmsf_csv, fig_dir / "rmsf.png",
            highlight_resids=[204, 392, 393, 394]),
    }
    expected = yaml.safe_load((Path(__file__).parent / "expected.yaml").read_text())
    summary = json.loads(mmpbsa_summary.json_path.read_text())
    occupancy = json.loads(analysis_results[0].occupancy_json.read_text()) \
        if analysis_results[0].occupancy_json else None
    report = figures.write_report(
        out_root / "report.md",
        expected=expected, mmpbsa_summary=summary, occupancy=occupancy,
    )
    return {"report": str(report), "figures": {k: str(v) for k, v in figs.items()}}


def cpu_smoke(
    config_path: Path,
    *,
    steps: int = 5,
    minimize_iterations: int = 200,
    timestep_fs: float = 0.1,
    temperature_K: float = 10.0,
    platform_name: str = "CPU",
) -> dict:
    """Run a short protein+ligand MD sanity check without membrane embedding."""

    from openmm import LangevinMiddleIntegrator, Platform, unit
    from openmm.app import DCDReporter, HBonds, NoCutoff, PDBFile, Simulation, StateDataReporter

    cfg = yaml.safe_load(config_path.read_text())
    paths = cfg["paths"]
    out_root = _abspath(paths["out_dir"]) / "cpu_smoke"
    out_root.mkdir(parents=True, exist_ok=True)

    lig = ligand.load_ligand(
        cfg["ligand"]["name"],
        cfg["ligand"]["smiles"],
        _abspath(paths["ligand_dir"]),
    )
    fetched = fetch.fetch_pdb(cfg["paper"]["pdb_bound_tmd"], _abspath(paths["structure_cache"]))
    fixed = prepare.fix_pdb(
        fetched.path,
        out_root / f"{cfg['paper']['pdb_bound_tmd']}_fixed_keep_ligand.pdb",
        keep_heterogens=True,
    )

    pdb = PDBFile(str(fixed.pdb_path))
    forcefield = make_forcefield(
        ("amber14/protein.ff14SB.xml", "amber14/tip3p.xml"),
        ligand_sdf=lig.sdf_path,
        ligand_ff_xml=lig.ff_xml,
    )
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=NoCutoff,
        constraints=HBonds,
        rigidWater=True,
    )
    integrator = LangevinMiddleIntegrator(
        temperature_K * unit.kelvin,
        1.0 / unit.picosecond,
        timestep_fs * unit.femtosecond,
    )
    integrator.setRandomNumberSeed(7)

    sim = Simulation(
        pdb.topology,
        system,
        integrator,
        Platform.getPlatformByName(platform_name),
    )
    sim.context.setPositions(pdb.positions)
    if minimize_iterations > 0:
        sim.minimizeEnergy(maxIterations=minimize_iterations)
    sim.context.setVelocitiesToTemperature(temperature_K * unit.kelvin, 7)

    traj_path = out_root / "bound_cpu_smoke.dcd"
    log_path = out_root / "bound_cpu_smoke.log"
    sim.reporters.append(DCDReporter(str(traj_path), 1))
    sim.reporters.append(
        StateDataReporter(
            str(log_path),
            max(1, steps // 4),
            step=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
        )
    )
    sim.step(steps)

    state = sim.context.getState(getPositions=True, getEnergy=True)
    topo_path = out_root / "bound_cpu_smoke_topology.pdb"
    with topo_path.open("w") as f:
        PDBFile.writeFile(pdb.topology, state.getPositions(), f, keepIds=True)

    analysis_result = analyze.run_analysis(
        analyze.AnalysisConfig(
            topology_pdb=topo_path,
            trajectory_dcd=traj_path,
            out_dir=out_root / "analysis",
            distances=[analyze.DistanceSpec(**d) for d in cfg["distances"]],
        )
    )

    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    return {
        "fixed_pdb": str(fixed.pdb_path),
        "topology": str(topo_path),
        "trajectory": str(traj_path),
        "log": str(log_path),
        "analysis": {
            "rmsd": str(analysis_result.rmsd_csv),
            "rmsf": str(analysis_result.rmsf_csv),
            "distances": str(analysis_result.distances_csv),
            "occupancy": str(analysis_result.occupancy_json),
        },
        "platform": platform_name,
        "steps": steps,
        "timestep_fs": timestep_fs,
        "temperature_K": temperature_K,
        "n_atoms": system.getNumParticles(),
        "final_potential_energy_kj_mol": energy,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="molgent-clc2",
                                     description="Reproduce Ma et al. 2023 (ClC-2/AK-42).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument(
        "--config", "-c", type=Path,
        default=Path(__file__).with_name("config.yaml"),
    )
    run_p.add_argument("--verbose", "-v", action="store_true")
    smoke_p = sub.add_parser("cpu-smoke")
    smoke_p.add_argument(
        "--config", "-c", type=Path,
        default=Path(__file__).with_name("config.yaml"),
    )
    smoke_p.add_argument("--steps", type=int, default=5)
    smoke_p.add_argument("--minimize-iterations", type=int, default=200)
    smoke_p.add_argument("--timestep-fs", type=float, default=0.1)
    smoke_p.add_argument("--temperature-k", type=float, default=10.0)
    smoke_p.add_argument("--platform", default="CPU")
    smoke_p.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if args.cmd == "cpu-smoke":
        out = cpu_smoke(
            args.config,
            steps=args.steps,
            minimize_iterations=args.minimize_iterations,
            timestep_fs=args.timestep_fs,
            temperature_K=args.temperature_k,
            platform_name=args.platform,
        )
    else:
        out = reproduce(args.config)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
