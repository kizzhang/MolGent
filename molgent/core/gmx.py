"""GROMACS subprocess wrappers for the paper-faithful (CHARMM36m) pipeline.

Pipeline driven from a CHARMM-GUI Membrane Builder bundle. CHARMM-GUI is the
only realistic way to assemble a ClC-2 + AK-42 + POPC + ions system that grompp
will accept, and the paper itself uses CHARMM-GUI as the system builder. The
wrapper here is intentionally thin: it validates the bundle, renders MDP
templates from :mod:`molgent.templates.gmx_mdp`, and shells out to ``gmx``.

Stages mirror the paper protocol:

* :func:`make_system` — locate ``step5_input.gro`` / ``topol.top`` /
  ``index.ndx`` inside the CHARMM-GUI bundle and stage them under ``out_dir``.
* :func:`minimize` — steepest descent (5000 steps).
* :func:`equilibrate_nvt` — 100 ps NVT, V-rescale, position-restrained.
* :func:`equilibrate_npt` — 1 ns NPT, Parrinello-Rahman semi-isotropic.
* :func:`production` — 5 ns × N replicates, no restraints.

CPU verification mode shrinks the durations (set via ``GmxConfig`` fields) so
that the whole pipeline can finish on a laptop without GPU support.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from importlib import resources
from pathlib import Path

log = logging.getLogger(__name__)


MDP_TEMPLATES_PACKAGE = "molgent.templates.gmx_mdp"


# --------------------------------------------------------------------------- #
# Data records
# --------------------------------------------------------------------------- #

@dataclass
class SystemFiles:
    """Paths produced by :func:`make_system`. All under one ``out_dir``."""

    gro: Path
    top: Path
    ndx: Path
    toppar_dir: Path | None = None  # CHARMM-GUI's toppar/ (FF + ligand .itp)


@dataclass
class StageResult:
    """Outputs of one MD stage (EM / NVT / NPT / production replicate)."""

    name: str
    gro: Path             # final coordinates
    cpt: Path | None      # checkpoint (None for EM)
    edr: Path             # energy file
    log: Path             # gmx log
    xtc: Path | None      # compressed trajectory (None for EM)
    tpr: Path             # binary run input


@dataclass
class GmxConfig:
    """Runtime parameters for the GROMACS pipeline.

    Most defaults match the paper. CPU-verify mode is realised by overriding
    ``nvt_ps``, ``npt_ps``, ``production_ns`` and ``n_replicas`` from the
    ``cpu_verify`` block in ``config.yaml``.
    """

    out_dir: Path
    name: str = "system"
    temperature_k: float = 310.0
    pressure_bar: float = 1.0
    timestep_fs: float = 2.0
    nvt_ps: float = 100.0
    npt_ps: float = 1000.0
    production_ns: float = 5.0
    n_replicas: int = 5
    write_interval_ps: float = 10.0
    seed_base: int = 1
    nt: int = 0                       # gmx mdrun -nt; 0 = auto
    pin: str = "auto"                 # gmx mdrun -pin
    extra_mdrun_args: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# subprocess plumbing
# --------------------------------------------------------------------------- #

def _gmx_binary() -> str:
    bin_path = shutil.which("gmx") or shutil.which("gmx_mpi")
    if bin_path is None:
        raise RuntimeError(
            "GROMACS not found on PATH (tried 'gmx', 'gmx_mpi'). "
            "Install via `brew install gromacs` (macOS) or your package manager."
        )
    return bin_path


def _run(
    args: list[str],
    *,
    cwd: Path,
    log_path: Path,
    stdin: str | None = None,
    timeout: float | None = None,
) -> None:
    """Run a command, tee stdout+stderr into ``log_path``, raise on non-zero."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("[gmx] %s  (cwd=%s)", " ".join(str(a) for a in args), cwd)
    with log_path.open("w") as f:
        f.write(f"# {' '.join(str(a) for a in args)}\n")
        f.flush()
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            input=stdin,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout,
        )
    if proc.returncode != 0:
        tail = log_path.read_text().splitlines()[-40:]
        raise RuntimeError(
            f"Command failed (rc={proc.returncode}): {' '.join(map(str, args))}\n"
            f"--- log tail ({log_path}) ---\n" + "\n".join(tail)
        )


# --------------------------------------------------------------------------- #
# MDP template rendering
# --------------------------------------------------------------------------- #

def _read_mdp_template(name: str) -> str:
    """Load a packaged MDP template (e.g. ``em.mdp``, ``nvt.mdp``)."""

    with resources.files(MDP_TEMPLATES_PACKAGE).joinpath(name).open() as f:
        return f.read()


def _render_mdp(name: str, out_path: Path, **params: object) -> Path:
    """Format placeholders in a template and write it out."""

    template = _read_mdp_template(name)
    try:
        rendered = template.format(**params)
    except KeyError as exc:
        raise KeyError(
            f"MDP template {name} expects placeholder {exc.args[0]!r}, not provided"
        ) from exc
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered)
    return out_path


def _ps_to_steps(ps: float, dt_fs: float) -> int:
    return int(round(ps * 1000.0 / dt_fs))


def _ns_to_steps(ns: float, dt_fs: float) -> int:
    return _ps_to_steps(ns * 1000.0, dt_fs)


# --------------------------------------------------------------------------- #
# CHARMM-GUI bundle handling
# --------------------------------------------------------------------------- #

def _find_gromacs_subdir(charmm_gui_dir: Path) -> Path:
    """Resolve the actual ``gromacs/`` directory inside a CHARMM-GUI bundle.

    CHARMM-GUI ships archives like ``charmm-gui-1234567890/`` with a
    ``gromacs/`` subfolder containing ``step5_input.gro``. Users sometimes
    extract one level too deep; we tolerate both layouts.
    """

    candidates = [
        charmm_gui_dir / "gromacs",
        charmm_gui_dir,
    ]
    candidates += list(charmm_gui_dir.glob("charmm-gui*/gromacs"))
    for c in candidates:
        if (c / "step5_input.gro").exists() and (c / "topol.top").exists():
            return c
    raise FileNotFoundError(
        f"Could not locate CHARMM-GUI gromacs/ output under {charmm_gui_dir}. "
        "Expected step5_input.gro + topol.top. Make sure the Membrane Builder "
        "archive has been extracted."
    )


def make_system(charmm_gui_dir: Path, out_dir: Path, *, name: str = "system") -> SystemFiles:
    """Stage CHARMM-GUI's grompp-ready bundle into ``out_dir``.

    We deliberately do **not** modify the topology — CHARMM-GUI is the
    canonical assembler and any rewriting risks losing toppar/ includes.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = _find_gromacs_subdir(Path(charmm_gui_dir))

    gro_in = src / "step5_input.gro"
    top_in = src / "topol.top"
    ndx_in = src / "index.ndx"
    toppar_in = src / "toppar"

    gro_out = out_dir / f"{name}.gro"
    top_out = out_dir / f"{name}.top"
    ndx_out = out_dir / f"{name}.ndx"

    shutil.copy2(gro_in, gro_out)
    shutil.copy2(top_in, top_out)

    if not ndx_in.exists():
        raise FileNotFoundError(
            f"CHARMM-GUI bundle at {src} is missing index.ndx — re-run the "
            "Membrane Builder with the GROMACS output checkbox enabled."
        )
    shutil.copy2(ndx_in, ndx_out)

    toppar_out: Path | None = None
    if toppar_in.exists():
        toppar_out = out_dir / "toppar"
        if toppar_out.exists():
            shutil.rmtree(toppar_out)
        shutil.copytree(toppar_in, toppar_out)

    log.info("Staged CHARMM-GUI bundle: %s -> %s", src, out_dir)
    return SystemFiles(gro=gro_out, top=top_out, ndx=ndx_out, toppar_dir=toppar_out)


# --------------------------------------------------------------------------- #
# MD stage runners
# --------------------------------------------------------------------------- #

def _grompp(
    *,
    mdp: Path,
    gro: Path,
    top: Path,
    ndx: Path,
    out_tpr: Path,
    cwd: Path,
    log_path: Path,
    restraint_gro: Path | None = None,
    maxwarn: int = 1,
) -> None:
    gmx = _gmx_binary()
    args = [
        gmx, "grompp",
        "-f", str(mdp),
        "-c", str(gro),
        "-p", str(top),
        "-n", str(ndx),
        "-o", str(out_tpr),
        "-maxwarn", str(maxwarn),
    ]
    if restraint_gro is not None:
        args.extend(["-r", str(restraint_gro)])
    _run(args, cwd=cwd, log_path=log_path)


def _mdrun(
    *,
    deffnm: str,
    cwd: Path,
    log_path: Path,
    cfg: GmxConfig,
) -> None:
    gmx = _gmx_binary()
    args = [gmx, "mdrun", "-deffnm", deffnm]
    if cfg.nt > 0:
        args.extend(["-nt", str(cfg.nt)])
    if cfg.pin:
        args.extend(["-pin", cfg.pin])
    args.extend(cfg.extra_mdrun_args)
    _run(args, cwd=cwd, log_path=log_path, timeout=None)


def minimize(system: SystemFiles, cfg: GmxConfig) -> StageResult:
    """Steepest-descent minimization (5000 steps, Fmax ~ 1000 kJ/mol/nm)."""

    stage_dir = cfg.out_dir / "em"
    stage_dir.mkdir(parents=True, exist_ok=True)
    mdp = stage_dir / "em.mdp"
    # em.mdp has no placeholders — copy directly
    mdp.write_text(_read_mdp_template("em.mdp"))

    deffnm = "em"
    tpr = stage_dir / f"{deffnm}.tpr"
    _grompp(
        mdp=mdp, gro=system.gro, top=system.top, ndx=system.ndx,
        out_tpr=tpr, cwd=stage_dir, log_path=stage_dir / "grompp.log",
        restraint_gro=system.gro,
    )
    _mdrun(deffnm=deffnm, cwd=stage_dir, log_path=stage_dir / "mdrun.log", cfg=cfg)

    return StageResult(
        name=deffnm,
        gro=stage_dir / f"{deffnm}.gro",
        cpt=None,
        edr=stage_dir / f"{deffnm}.edr",
        log=stage_dir / f"{deffnm}.log",
        xtc=None,
        tpr=tpr,
    )


def equilibrate_nvt(
    system: SystemFiles,
    em: StageResult,
    cfg: GmxConfig,
) -> StageResult:
    """NVT equilibration with position restraints (paper: 100 ps)."""

    stage_dir = cfg.out_dir / "nvt"
    stage_dir.mkdir(parents=True, exist_ok=True)
    nsteps = _ps_to_steps(cfg.nvt_ps, cfg.timestep_fs)
    mdp = _render_mdp(
        "nvt.mdp",
        stage_dir / "nvt.mdp",
        timestep_ps=f"{cfg.timestep_fs / 1000.0:.4f}",
        nsteps=nsteps,
        temperature_k=f"{cfg.temperature_k:.2f}",
        seed=cfg.seed_base,
    )

    deffnm = "nvt"
    tpr = stage_dir / f"{deffnm}.tpr"
    _grompp(
        mdp=mdp, gro=em.gro, top=system.top, ndx=system.ndx,
        out_tpr=tpr, cwd=stage_dir, log_path=stage_dir / "grompp.log",
        restraint_gro=em.gro,
    )
    _mdrun(deffnm=deffnm, cwd=stage_dir, log_path=stage_dir / "mdrun.log", cfg=cfg)

    return StageResult(
        name=deffnm,
        gro=stage_dir / f"{deffnm}.gro",
        cpt=stage_dir / f"{deffnm}.cpt",
        edr=stage_dir / f"{deffnm}.edr",
        log=stage_dir / f"{deffnm}.log",
        xtc=stage_dir / f"{deffnm}.xtc",
        tpr=tpr,
    )


def equilibrate_npt(
    system: SystemFiles,
    nvt: StageResult,
    cfg: GmxConfig,
) -> StageResult:
    """NPT equilibration with Parrinello-Rahman semi-isotropic (paper: 1 ns)."""

    stage_dir = cfg.out_dir / "npt"
    stage_dir.mkdir(parents=True, exist_ok=True)
    nsteps = _ps_to_steps(cfg.npt_ps, cfg.timestep_fs)
    mdp = _render_mdp(
        "npt.mdp",
        stage_dir / "npt.mdp",
        timestep_ps=f"{cfg.timestep_fs / 1000.0:.4f}",
        nsteps=nsteps,
        temperature_k=f"{cfg.temperature_k:.2f}",
        pressure_bar=f"{cfg.pressure_bar:.3f}",
    )

    deffnm = "npt"
    tpr = stage_dir / f"{deffnm}.tpr"
    _grompp(
        mdp=mdp, gro=nvt.gro, top=system.top, ndx=system.ndx,
        out_tpr=tpr, cwd=stage_dir, log_path=stage_dir / "grompp.log",
        restraint_gro=nvt.gro,
    )
    _mdrun(deffnm=deffnm, cwd=stage_dir, log_path=stage_dir / "mdrun.log", cfg=cfg)

    return StageResult(
        name=deffnm,
        gro=stage_dir / f"{deffnm}.gro",
        cpt=stage_dir / f"{deffnm}.cpt",
        edr=stage_dir / f"{deffnm}.edr",
        log=stage_dir / f"{deffnm}.log",
        xtc=stage_dir / f"{deffnm}.xtc",
        tpr=tpr,
    )


def production(
    system: SystemFiles,
    npt: StageResult,
    cfg: GmxConfig,
) -> list[StageResult]:
    """Run ``cfg.n_replicas`` production replicates seeded from ``cfg.seed_base``."""

    write_steps = _ps_to_steps(cfg.write_interval_ps, cfg.timestep_fs)
    nsteps = _ns_to_steps(cfg.production_ns, cfg.timestep_fs)
    nstenergy = max(1, write_steps // 10)
    nstlog = nstenergy

    results: list[StageResult] = []
    for rep in range(cfg.n_replicas):
        rep_dir = cfg.out_dir / "prod" / f"rep{rep}"
        rep_dir.mkdir(parents=True, exist_ok=True)
        mdp = _render_mdp(
            "prod.mdp",
            rep_dir / "prod.mdp",
            timestep_ps=f"{cfg.timestep_fs / 1000.0:.4f}",
            nsteps=nsteps,
            temperature_k=f"{cfg.temperature_k:.2f}",
            pressure_bar=f"{cfg.pressure_bar:.3f}",
            nstenergy=nstenergy,
            nstlog=nstlog,
            nstxout_compressed=write_steps,
        )

        deffnm = "prod"
        tpr = rep_dir / f"{deffnm}.tpr"
        _grompp(
            mdp=mdp, gro=npt.gro, top=system.top, ndx=system.ndx,
            out_tpr=tpr, cwd=rep_dir, log_path=rep_dir / "grompp.log",
        )

        rep_cfg = GmxConfig(**{**asdict(cfg), "out_dir": rep_dir,
                                "seed_base": cfg.seed_base + rep})
        _mdrun(deffnm=deffnm, cwd=rep_dir, log_path=rep_dir / "mdrun.log", cfg=rep_cfg)

        results.append(
            StageResult(
                name=f"{deffnm}_rep{rep}",
                gro=rep_dir / f"{deffnm}.gro",
                cpt=rep_dir / f"{deffnm}.cpt",
                edr=rep_dir / f"{deffnm}.edr",
                log=rep_dir / f"{deffnm}.log",
                xtc=rep_dir / f"{deffnm}.xtc",
                tpr=tpr,
            )
        )

    summary = cfg.out_dir / "prod" / "summary.json"
    summary.write_text(json.dumps(
        {"replicas": [r.name for r in results],
         "production_ns": cfg.production_ns,
         "write_interval_ps": cfg.write_interval_ps},
        indent=2,
    ))
    return results


def run_full(
    charmm_gui_dir: Path,
    cfg: GmxConfig,
) -> dict[str, object]:
    """End-to-end orchestrator: stage bundle, EM, NVT, NPT, production."""

    system = make_system(charmm_gui_dir, cfg.out_dir, name=cfg.name)
    em = minimize(system, cfg)
    nvt = equilibrate_nvt(system, em, cfg)
    npt = equilibrate_npt(system, nvt, cfg)
    prod = production(system, npt, cfg)
    return {"system": system, "em": em, "nvt": nvt, "npt": npt, "prod": prod}
