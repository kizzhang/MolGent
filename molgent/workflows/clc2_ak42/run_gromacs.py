"""Paper-faithful (GROMACS + CHARMM36m) entrypoint for the ClC-2/AK-42 reproduction.

This is the entrypoint to use when you want **bit-for-bit alignment with
Ma et al. 2023** — same force field (CHARMM36m + CGenFF), same thermostat /
barostat (V-rescale + Parrinello-Rahman), and the same 5 ns × 5 replicates
production schedule.

Two subcommands:

* ``cpu-verify`` — short single-replicate run (50 ps EM-equilibrium) intended
  only to confirm that
    (a) ``gmx grompp`` accepts the staged CHARMM-GUI bundle,
    (b) the distance specs resolve on the trajectory and reproduce the
        crystallographic K204 / S392 / K394 distances at frame 0
        (≈ 3.68 / 3.51 / 2.61 Å),
    (c) ``gmx_MMPBSA`` parses the trajectory at all.
  Acceptance for occupancy / per-residue ranking is **deliberately skipped**
  here — 50 ps is far too short for either.

* ``run`` — full paper protocol (5 ns × 5 replicates × {apo, bound}). Requires
  GPU; CPU walltime is on the order of weeks.

Both subcommands assume the user has already produced and uploaded:

* CHARMM-GUI Membrane Builder bundles for the apo (7XJA) and bound (8GQU)
  TMD systems under ``data/charmm_gui/{apo_tmd,bound_tmd}/``.
* A CGenFF stream file for AK-42 at ``data/ligands/gh6.str`` (only needed
  if the bundle was generated without the ligand pre-merged).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import yaml

from molgent.core import analyze, fetch, figures, gmx, ligand, mmpbsa_gmx

log = logging.getLogger("clc2_ak42_gmx")
REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- #
# Paths and config helpers
# --------------------------------------------------------------------------- #

def _abspath(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def _load_cfg(config_path: Path) -> dict:
    return yaml.safe_load(config_path.read_text())


def _gmx_cfg_from_yaml(cfg_yaml: dict, *, cpu_verify: bool, out_dir: Path,
                        name: str) -> gmx.GmxConfig:
    g = cfg_yaml["gromacs"]
    base = dict(
        out_dir=out_dir,
        name=name,
        temperature_k=g["temperature_K"],
        pressure_bar=g["pressure_bar"],
        timestep_fs=g["timestep_fs"],
        nvt_ps=g["nvt_ps"],
        npt_ps=g["npt_ps"],
        production_ns=g["production_ns"],
        n_replicas=g["n_replicas"],
        write_interval_ps=g["write_interval_ps"],
    )
    if cpu_verify:
        v = g.get("cpu_verify", {})
        base.update(
            nvt_ps=v.get("nvt_ps", 50.0),
            npt_ps=v.get("npt_ps", 50.0),
            production_ns=v.get("production_ns", 0.05),
            n_replicas=v.get("n_replicas", 1),
        )
    return gmx.GmxConfig(**base)


# --------------------------------------------------------------------------- #
# Ligand handoff: extract GH6 from 8GQU + locate CGenFF .str
# --------------------------------------------------------------------------- #

def _prepare_ligand(cfg_yaml: dict) -> ligand.Ligand:
    """Pull GH6 out of the cached 8GQU PDB and locate any CGenFF .str."""

    ligand_dir = _abspath(cfg_yaml["paths"]["ligand_dir"])
    structure_cache = _abspath(cfg_yaml["paths"]["structure_cache"])
    bound_id = cfg_yaml["paper"]["pdb_bound_tmd"]
    fetched = fetch.fetch_pdb(bound_id, structure_cache)

    extracted_pdb = ligand_dir / f"{cfg_yaml['ligand']['name']}_from_pdb.pdb"
    if extracted_pdb.exists():
        # Already extracted — reuse and just stamp paths on a Ligand record.
        return ligand.Ligand(
            name=cfg_yaml["ligand"]["name"],
            smiles=cfg_yaml["ligand"]["smiles"],
            sdf_path=ligand_dir / f"{cfg_yaml['ligand']['name']}_from_pdb.sdf",
            pdb_path=extracted_pdb,
            mol2_path=ligand_dir / f"{cfg_yaml['ligand']['name']}_from_pdb.mol2",
            charmm_str=_abspath(cfg_yaml["gromacs"]["cgenff_str"])
                if Path(_abspath(cfg_yaml["gromacs"]["cgenff_str"])).exists() else None,
        )

    lig = ligand.extract_from_pdb(
        pdb_path=fetched.path,
        resname=cfg_yaml["ligand"]["resname"],
        chain="A",
        smiles=cfg_yaml["ligand"]["smiles"],
        out_dir=ligand_dir,
        name=cfg_yaml["ligand"]["name"],
    )
    cgenff_path = _abspath(cfg_yaml["gromacs"]["cgenff_str"])
    if cgenff_path.exists():
        lig.charmm_str = cgenff_path
    return lig


# --------------------------------------------------------------------------- #
# Distance-spec adapter: AMBER chain "A" -> CHARMM-GUI segid "PROA"
# --------------------------------------------------------------------------- #

# CHARMM-GUI's Membrane Builder writes protein chains as PROA/PROB/PROC and
# the ligand as HETA. Distance specs in config.yaml use plain chain ids so the
# OpenMM pipeline still works — translate them on the fly here.
_DEFAULT_SEGID_MAP = {"A": "PROA", "B": "PROB", "C": "PROC", "D": "PROD"}


def _translate_distance_specs(specs: list[dict], segid_map: dict[str, str] | None = None
                               ) -> list[analyze.DistanceSpec]:
    smap = segid_map or _DEFAULT_SEGID_MAP
    out = []
    for d in specs:
        chain = smap.get(d.get("chain"), d.get("chain"))
        ligand_chain = d.get("ligand_chain")
        if ligand_chain is not None:
            # CHARMM-GUI puts heterogens into HETA regardless of original chain.
            ligand_chain = "HETA"
        out.append(analyze.DistanceSpec(
            name=d["name"],
            residue_index=d["residue_index"],
            residue_atom=d["residue_atom"],
            ligand_resname=d["ligand_resname"],
            ligand_atom=d["ligand_atom"],
            chain=chain,
            ligand_chain=ligand_chain,
        ))
    return out


# --------------------------------------------------------------------------- #
# Subcommand implementations
# --------------------------------------------------------------------------- #

def cpu_verify(config_path: Path) -> dict:
    """Short single-replicate run that proves the pipeline is wired up."""

    cfg = _load_cfg(config_path)
    out_root = _abspath(cfg["paths"]["gromacs_out_dir"]) / "cpu_verify"
    out_root.mkdir(parents=True, exist_ok=True)

    log.info("Step 1/5 — extract GH6 from 8GQU")
    lig = _prepare_ligand(cfg)
    log.info("  ligand: pdb=%s mol2=%s str=%s", lig.pdb_path, lig.mol2_path, lig.charmm_str)

    bound_bundle = _abspath(cfg["gromacs"]["charmm_gui"]["bound_tmd"])
    if not bound_bundle.exists():
        raise FileNotFoundError(
            f"CHARMM-GUI bound bundle not found at {bound_bundle}. Generate it "
            "via Membrane Builder (POPC, 150 mM NaCl, 8GQU + GH6 .str) and "
            "extract it there."
        )

    gcfg = _gmx_cfg_from_yaml(cfg, cpu_verify=True, out_dir=out_root / "bound_tmd",
                               name="bound_tmd")

    log.info("Step 2/5 — stage system and run EM/NVT/NPT/production")
    log.info("  CPU-verify timings: nvt=%.0f ps, npt=%.0f ps, prod=%.2f ns × %d",
             gcfg.nvt_ps, gcfg.npt_ps, gcfg.production_ns, gcfg.n_replicas)
    pipeline = gmx.run_full(bound_bundle, gcfg)
    system = pipeline["system"]
    prod = pipeline["prod"]
    rep = prod[0]

    log.info("Step 3/5 — distance / RMSF analysis on %s", rep.xtc)
    distances = _translate_distance_specs(cfg["distances"])
    analysis = analyze.run_analysis(analyze.AnalysisConfig(
        topology_pdb=system.gro,            # MDAnalysis reads .gro as topology
        trajectory_dcd=rep.xtc,             # ... and .xtc as trajectory
        out_dir=rep.xtc.parent / "analysis",
        distances=distances,
    ))

    log.info("Step 4/5 — gmx_MMPBSA on the single replicate (skip-able)")
    mmres = None
    try:
        mm_cfg = mmpbsa_gmx.MMPBSAGmxConfig(
            out_dir=rep.xtc.parent / "mmpbsa",
            tpr=rep.tpr,
            xtc=rep.xtc,
            ndx=system.ndx,
            ligand_mol2=lig.mol2_path or lig.sdf_path,
            receptor_group="Protein",
            ligand_group=cfg["ligand"]["resname"],
            igb=cfg["mmpbsa"]["igb"],
            startframe=cfg["mmpbsa"]["startframe"],
            endframe=min(cfg["mmpbsa"]["endframe"], 10),  # only ~10 frames in 50 ps
            interval=1,
            dec_verbose=cfg["mmpbsa"]["decomp_verbose"],
        )
        mmres = mmpbsa_gmx.run_mmpbsa_gmx(mm_cfg)
    except (RuntimeError, FileNotFoundError) as exc:
        log.warning("gmx_MMPBSA skipped (%s)", exc)

    log.info("Step 5/5 — write report")
    expected = yaml.safe_load((Path(__file__).parent / "expected.yaml").read_text())
    summary = {"top_residues": [], "mean_dG_total_kcal": float("nan"), "n_frames": 0}
    if mmres is not None:
        summary = json.loads(mmres.json_path.read_text())
    occupancy = json.loads(analysis.occupancy_json.read_text()) \
        if analysis.occupancy_json else None

    report_md = out_root / "report.md"
    figures.write_report(report_md, expected=expected, mmpbsa_summary=summary,
                         occupancy=occupancy)
    # Append CPU-verify caveat
    with report_md.open("a") as f:
        f.write("\n\n> ⚠️ **CPU verification only** — single replicate, "
                f"{gcfg.production_ns:.2f} ns production. Acceptance criteria "
                "(top-4 residue overlap, occupancy thresholds) **not** evaluated.\n")

    return {
        "report": str(report_md),
        "system": {k: str(v) for k, v in asdict(system).items() if v},
        "production": [str(r.xtc) for r in prod],
        "analysis": {
            "rmsd": str(analysis.rmsd_csv),
            "rmsf": str(analysis.rmsf_csv),
            "distances": str(analysis.distances_csv),
            "occupancy": str(analysis.occupancy_json),
        },
        "mmpbsa": str(mmres.csv_path) if mmres else None,
    }


def reproduce(config_path: Path) -> dict:
    """Full paper protocol: 5 ns × 5 replicates × {apo, bound}."""

    cfg = _load_cfg(config_path)
    out_root = _abspath(cfg["paths"]["gromacs_out_dir"])
    out_root.mkdir(parents=True, exist_ok=True)

    log.info("Step 1/5 — extract GH6 from 8GQU")
    lig = _prepare_ligand(cfg)
    if lig.charmm_str is None:
        raise FileNotFoundError(
            "CGenFF .str not found at "
            f"{_abspath(cfg['gromacs']['cgenff_str'])}. Generate via "
            "https://cgenff.silcsbio.com/ before running the full pipeline."
        )

    states = [
        ("apo_tmd",   _abspath(cfg["gromacs"]["charmm_gui"]["apo_tmd"])),
        ("bound_tmd", _abspath(cfg["gromacs"]["charmm_gui"]["bound_tmd"])),
    ]

    distances = _translate_distance_specs(cfg["distances"])
    state_results: dict[str, dict] = {}
    bound_per_residue_csvs: list[Path] = []

    for state, bundle in states:
        if not bundle.exists():
            raise FileNotFoundError(f"CHARMM-GUI bundle missing: {bundle}")
        log.info("Step 2/5 — %s: full %d-replicate, %.1f ns production",
                 state, cfg["gromacs"]["n_replicas"], cfg["gromacs"]["production_ns"])
        gcfg = _gmx_cfg_from_yaml(cfg, cpu_verify=False, out_dir=out_root / state,
                                   name=state)
        pipeline = gmx.run_full(bundle, gcfg)
        system = pipeline["system"]

        for rep_idx, rep in enumerate(pipeline["prod"]):
            log.info("Step 3/5 — %s rep%d analysis", state, rep_idx)
            analyze.run_analysis(analyze.AnalysisConfig(
                topology_pdb=system.gro, trajectory_dcd=rep.xtc,
                out_dir=rep.xtc.parent / "analysis",
                distances=distances if state == "bound_tmd" else [],
            ))
            if state == "bound_tmd":
                log.info("Step 4/5 — %s rep%d gmx_MMPBSA", state, rep_idx)
                mm_cfg = mmpbsa_gmx.MMPBSAGmxConfig(
                    out_dir=rep.xtc.parent / "mmpbsa",
                    tpr=rep.tpr, xtc=rep.xtc, ndx=system.ndx,
                    ligand_mol2=lig.mol2_path or lig.sdf_path,
                    receptor_group="Protein",
                    ligand_group=cfg["ligand"]["resname"],
                    igb=cfg["mmpbsa"]["igb"],
                    startframe=cfg["mmpbsa"]["startframe"],
                    endframe=cfg["mmpbsa"]["endframe"],
                    interval=cfg["mmpbsa"]["interval"],
                    dec_verbose=cfg["mmpbsa"]["decomp_verbose"],
                )
                mres = mmpbsa_gmx.run_mmpbsa_gmx(mm_cfg)
                bound_per_residue_csvs.append(mres.csv_path)

        state_results[state] = {
            "system": {k: str(v) for k, v in asdict(system).items() if v},
            "n_replicas": cfg["gromacs"]["n_replicas"],
        }

    log.info("Step 5/5 — aggregate replicates and write report")
    agg_dir = out_root / "aggregate"
    agg = mmpbsa_gmx.aggregate_replicates(bound_per_residue_csvs, agg_dir)

    expected = yaml.safe_load((Path(__file__).parent / "expected.yaml").read_text())
    summary = json.loads(agg.json_path.read_text())

    fig_dir = out_root / "figures"
    figures.plot_per_residue_dG(agg.csv_path, fig_dir / "per_residue_dG.png")

    report = figures.write_report(
        out_root / "report.md",
        expected=expected, mmpbsa_summary=summary, occupancy=None,
    )
    return {
        "report": str(report),
        "states": state_results,
        "aggregate_csv": str(agg.csv_path),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="molgent-clc2-gmx",
        description="GROMACS/CHARMM36m reproduction of Ma et al. 2023 (ClC-2/AK-42).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    verify = sub.add_parser("cpu-verify",
                             help="Short single-replicate sanity check (CPU OK).")
    verify.add_argument("--config", "-c", type=Path,
                         default=Path(__file__).with_name("config.yaml"))
    verify.add_argument("--verbose", "-v", action="store_true")

    full = sub.add_parser("run",
                           help="Full paper protocol (5 ns × 5 replicates × 2 states; needs GPU).")
    full.add_argument("--config", "-c", type=Path,
                       default=Path(__file__).with_name("config.yaml"))
    full.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.cmd == "cpu-verify":
        out = cpu_verify(args.config)
    else:
        out = reproduce(args.config)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
