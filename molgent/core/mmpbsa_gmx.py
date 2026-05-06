"""Per-residue MM/GBSA decomposition via gmx_MMPBSA (paper protocol).

This is the GROMACS-side counterpart to :mod:`molgent.core.mmpbsa`. The OpenMM
implementation does its own per-residue zeroing trick; here we delegate to the
canonical AmberTools workflow that the paper itself uses (Ma et al. 2023 cite
the MM/PBSA & MM/GBSA review whose practical realisation in modern Gromacs is
``gmx_MMPBSA``).

Inputs (per replicate):

* ``tpr`` — production TPR (any state file with topology works; production is
  the default because it carries the trajectory's box).
* ``xtc`` — production trajectory.
* ``ndx`` — index file containing receptor and ligand groups.
* ``ligand_mol2`` — MOL2 of the ligand (RDKit export from
  :func:`molgent.core.ligand.extract_from_pdb`); needed by gmx_MMPBSA to
  re-build a GAFF parameterisation for energy decomposition.

Outputs (mirror :mod:`molgent.core.mmpbsa` on purpose so figures/report code
keeps working):

* ``per_residue.csv`` with columns
  ``chain, residue_index, residue_name, dG_kcal_mol, std_kcal_mol``.
* ``summary.json`` with ``mean_dG_total_kcal``, ``n_frames``, ``top_residues``.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MDP_TEMPLATES_PACKAGE = "molgent.templates.gmx_mdp"


# --------------------------------------------------------------------------- #
# Data records
# --------------------------------------------------------------------------- #

@dataclass
class MMPBSAGmxConfig:
    out_dir: Path
    tpr: Path
    xtc: Path
    ndx: Path
    ligand_mol2: Path
    receptor_group: str = "Protein"
    ligand_group: str = "GH6"
    igb: int = 8                     # GBn2 (Onufriev)
    startframe: int = 1
    endframe: int = 500
    interval: int = 5
    dec_verbose: int = 2             # idecomp=4, dec_verbose=2 (paper-style)
    extra_args: list[str] = field(default_factory=list)


@dataclass
class MMPBSAGmxResult:
    csv_path: Path
    json_path: Path
    raw_decomp_path: Path | None
    raw_results_path: Path | None
    top_residues: list[tuple[str, float]]
    mean_dG_total_kcal: float


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _gmxmmpbsa_binary() -> str:
    bin_path = shutil.which("gmx_MMPBSA")
    if bin_path is None:
        raise RuntimeError(
            "gmx_MMPBSA not found on PATH. Install in an AmberTools environment: "
            "`mamba install -c conda-forge ambertools=23 parmed=4` then "
            "`pip install gmx_MMPBSA`."
        )
    return bin_path


def _render_input(cfg: MMPBSAGmxConfig, out_path: Path) -> Path:
    """Write the gmx_MMPBSA input file from the packaged template."""

    with resources.files(MDP_TEMPLATES_PACKAGE).joinpath("mmpbsa.in").open() as f:
        template = f.read()
    rendered = template.format(
        startframe=cfg.startframe,
        endframe=cfg.endframe,
        interval=cfg.interval,
        igb=cfg.igb,
        dec_verbose=cfg.dec_verbose,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered)
    return out_path


# Match canonical residue lines emitted by gmx_MMPBSA's per-residue decomp file.
# Format examples (column widths vary slightly across versions):
#   "R LYS  204     | ... | -2.44   0.60   0.04"
#   "R LYS 204:A    | ... | -2.44   0.60   0.04"
# We pick the *last* triple of floats on the line as (mean, sd, sem).
_RES_LINE = re.compile(
    r"""^[A-Z]\s+              # one-letter type tag (R = receptor, L = ligand)
        ([A-Z]{2,4})\s+        # 1: residue name
        (\d+)                  # 2: residue index
        (?::([A-Za-z0-9]))?    # 3: optional chain id
        \s*\|.*?               # skip middle columns
        (-?\d+\.\d+)\s+        # 4: TOTAL mean
        (-?\d+\.\d+)\s+        # 5: TOTAL std
        (-?\d+\.\d+)\s*$       # 6: TOTAL sem
    """,
    re.VERBOSE,
)


def _parse_decomp_file(path: Path) -> list[tuple[str, int, str, float, float]]:
    """Parse FINAL_DECOMP_MMPBSA.dat into (chain, resid, resname, mean, std)."""

    rows: list[tuple[str, int, str, float, float]] = []
    in_total_block = False
    text = path.read_text().splitlines()
    for line in text:
        # Heuristic: a "TOTAL" header signals start of the section we want;
        # other sections (Sidechain, Backbone) are filtered out.
        if "Total Energy Decomposition" in line:
            in_total_block = True
            continue
        if "Sidechain Energy Decomposition" in line or "Backbone Energy Decomposition" in line:
            in_total_block = False
            continue
        if not in_total_block:
            continue
        m = _RES_LINE.match(line.strip())
        if not m:
            continue
        resname, resid, chain, mean_s, std_s, _sem_s = m.groups()
        rows.append(
            (chain or "A", int(resid), resname, float(mean_s), float(std_s))
        )
    return rows


def _parse_results_total_kcal(path: Path) -> float | None:
    """Extract the binding free energy from FINAL_RESULTS_MMPBSA.dat (kcal/mol).

    gmx_MMPBSA writes a section like:
        Delta (Complex - Receptor - Ligand):
        DELTA TOTAL          -32.45   1.23
    """

    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("DELTA TOTAL"):
            parts = s.split()
            try:
                return float(parts[2])
            except (IndexError, ValueError):
                continue
    return None


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_mmpbsa_gmx(cfg: MMPBSAGmxConfig, top_k: int = 10) -> MMPBSAGmxResult:
    """Run gmx_MMPBSA and convert outputs to the project's CSV/JSON schema."""

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    in_path = _render_input(cfg, out / "mmpbsa.in")

    args = [
        _gmxmmpbsa_binary(), "-O",
        "-i", str(in_path),
        "-cs", str(cfg.tpr),
        "-ci", str(cfg.ndx),
        "-cg", cfg.receptor_group, cfg.ligand_group,
        "-ct", str(cfg.xtc),
        "-lm", str(cfg.ligand_mol2),
        "-eo", str(out / "energy_per_frame.csv"),
        "-deo", str(out / "energy_per_residue.csv"),
        *cfg.extra_args,
    ]

    log_path = out / "gmx_MMPBSA.log"
    log.info("Running gmx_MMPBSA (-> %s)", log_path)
    with log_path.open("w") as f:
        f.write("# " + " ".join(args) + "\n")
        f.flush()
        proc = subprocess.run(
            args, cwd=str(out), stdout=f, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text().splitlines()[-50:])
        raise RuntimeError(
            f"gmx_MMPBSA failed (rc={proc.returncode}). Tail:\n{tail}"
        )

    decomp_path = out / "FINAL_DECOMP_MMPBSA.dat"
    results_path = out / "FINAL_RESULTS_MMPBSA.dat"
    if not decomp_path.exists():
        raise FileNotFoundError(
            f"Expected gmx_MMPBSA output {decomp_path} missing; check {log_path}."
        )

    rows = _parse_decomp_file(decomp_path)
    if not rows:
        raise RuntimeError(
            f"Parsed 0 residues from {decomp_path}; gmx_MMPBSA may have used an "
            "unexpected output format. Inspect the file and adjust _RES_LINE."
        )

    csv_path = out / "per_residue.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chain", "residue_index", "residue_name", "dG_kcal_mol", "std_kcal_mol"])
        rows.sort(key=lambda r: abs(r[3]), reverse=True)
        w.writerows(rows)

    top = [(f"{name}{idx}", dg) for chain, idx, name, dg, _ in rows[:top_k]]
    mean_total = _parse_results_total_kcal(results_path)
    summary = {
        "mean_dG_total_kcal": mean_total if mean_total is not None else float("nan"),
        "n_frames": (cfg.endframe - cfg.startframe) // cfg.interval + 1,
        "top_residues": top,
    }
    json_path = out / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    return MMPBSAGmxResult(
        csv_path=csv_path,
        json_path=json_path,
        raw_decomp_path=decomp_path,
        raw_results_path=results_path if results_path.exists() else None,
        top_residues=top,
        mean_dG_total_kcal=summary["mean_dG_total_kcal"],
    )


def aggregate_replicates(
    per_residue_csvs: list[Path],
    out_dir: Path,
    *,
    top_k: int = 10,
) -> MMPBSAGmxResult:
    """Average per-residue ΔG across replicate runs and rewrite outputs.

    Each input CSV is the ``per_residue.csv`` from one replicate. Means are
    weighted equally (paper protocol: 5 × 5 ns replicates). The aggregated
    output uses the same CSV schema as the per-replicate files, so figures
    and report code work unchanged.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accum: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for csv_path in per_residue_csvs:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for r in reader:
                key = (r["chain"], int(r["residue_index"]), r["residue_name"])
                accum[key].append(float(r["dG_kcal_mol"]))

    rows = []
    for (chain, idx, name), values in accum.items():
        arr = np.asarray(values)
        rows.append((chain, idx, name, float(arr.mean()), float(arr.std(ddof=0))))
    rows.sort(key=lambda r: abs(r[3]), reverse=True)

    csv_path = out_dir / "per_residue.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chain", "residue_index", "residue_name", "dG_kcal_mol", "std_kcal_mol"])
        w.writerows(rows)

    top = [(f"{name}{idx}", dg) for chain, idx, name, dg, _ in rows[:top_k]]
    summary = {
        "mean_dG_total_kcal": float("nan"),  # not aggregated here
        "n_frames": None,
        "n_replicates": len(per_residue_csvs),
        "top_residues": top,
    }
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    return MMPBSAGmxResult(
        csv_path=csv_path,
        json_path=json_path,
        raw_decomp_path=None,
        raw_results_path=None,
        top_residues=top,
        mean_dG_total_kcal=summary["mean_dG_total_kcal"],
    )
