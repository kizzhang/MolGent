"""Reproduce the paper-style figures from MM/PBSA + analysis outputs."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

log = logging.getLogger(__name__)

PAPER_HIGHLIGHT = {"K204", "S392", "Q393", "K394"}


def _read_per_residue(csv_path: Path) -> list[tuple[str, int, str, float, float]]:
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((
                r["chain"],
                int(r["residue_index"]),
                r["residue_name"],
                float(r["dG_kcal_mol"]),
                float(r["std_kcal_mol"]),
            ))
    return rows


def plot_per_residue_dG(csv_path: Path, out_png: Path, top_n: int = 15) -> Path:
    """Bar chart of per-residue ΔG with paper-highlighted residues coloured."""

    import matplotlib.pyplot as plt

    rows = _read_per_residue(csv_path)
    rows.sort(key=lambda r: abs(r[3]), reverse=True)
    rows = rows[:top_n]

    labels = [f"{r[2]}{r[1]}" for r in rows]
    values = [r[3] for r in rows]
    errors = [r[4] for r in rows]
    colours = ["#d62728" if lbl in PAPER_HIGHLIGHT else "#1f77b4" for lbl in labels]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(labels)), values, yerr=errors, color=colours, capsize=3)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right")
    ax.set_ylabel(r"$\Delta G_{bind}$ contribution (kcal/mol)")
    ax.set_title("Per-residue contribution to AK-42 binding (top {})".format(top_n))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


def plot_distances(csv_path: Path, out_png: Path, cutoff_A: float = 3.5) -> Path:
    """Time-series plot of the diagnostic H-bond distances."""

    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.genfromtxt(csv_path, delimiter=",", names=True)
    names = [n for n in arr.dtype.names if n != "time_ps"]
    time_ns = arr["time_ps"] / 1000.0

    fig, ax = plt.subplots(figsize=(8, 4))
    for n in names:
        ax.plot(time_ns, arr[n] * 10.0, label=n, linewidth=0.8)  # nm -> Å
    ax.axhline(cutoff_A, color="grey", linestyle="--", linewidth=0.7, label=f"{cutoff_A} Å")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Distance (Å)")
    ax.set_title("Diagnostic H-bond distances")
    ax.legend(loc="upper right", fontsize=8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


def plot_rmsf(rmsf_csv: Path, out_png: Path, highlight_resids: list[int] | None = None) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.genfromtxt(rmsf_csv, delimiter=",", names=True, dtype=None, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(arr["residue_index"], arr["rmsf_A"], color="#1f77b4")
    if highlight_resids:
        for r in highlight_resids:
            ax.axvline(r, color="#d62728", alpha=0.5, linewidth=0.8)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("RMSF (Å)")
    ax.set_title("Per-residue RMSF")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return out_png


def write_report(
    out_md: Path,
    *,
    expected: dict,
    mmpbsa_summary: dict,
    occupancy: dict | None = None,
) -> Path:
    """Write a side-by-side paper-vs-reproduction report."""

    expected_top = set(expected.get("top_residues", []))
    reproduced_top = [name for name, _ in mmpbsa_summary.get("top_residues", [])][:10]
    overlap = expected_top.intersection(reproduced_top[:4])

    lines = ["# MolGent — ClC-2 / AK-42 reproduction report", ""]
    lines.append("## MM/GBSA per-residue summary")
    lines.append(f"- Mean ΔE_bind: **{mmpbsa_summary.get('mean_dG_total_kcal'):.2f} kcal/mol**")
    lines.append(f"- Frames analysed: {mmpbsa_summary.get('n_frames')}")
    lines.append("")
    lines.append("| Rank | Residue | ΔG (kcal/mol) | In paper top set? |")
    lines.append("|---:|---|---:|:---:|")
    for i, (name, dg) in enumerate(mmpbsa_summary.get("top_residues", [])[:10], 1):
        marker = "✅" if name in expected_top else " "
        lines.append(f"| {i} | {name} | {dg:.2f} | {marker} |")
    lines.append("")
    lines.append(f"Paper-reported top residues: {sorted(expected_top)}")
    lines.append(f"Reproduced top-4: {reproduced_top[:4]}")
    lines.append(f"Overlap with paper top set (target ≥ 3 of 4): **{len(overlap)} / {len(expected_top)}**")
    lines.append("")

    if occupancy is not None:
        lines.append("## H-bond occupancy")
        lines.append("| Pair | Occupancy (≤ 3.5 Å) |")
        lines.append("|---|---:|")
        for k, v in occupancy.items():
            lines.append(f"| {k} | {v:.1%} |")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")
    return out_md
