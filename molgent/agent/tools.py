"""Tool definitions exposed to the MolGent agent.

Each tool is a thin wrapper over a function in ``molgent.core.*``. The schema
follows the Anthropic SDK tool-use format. Dispatch is by tool name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from molgent.core import analyze, fetch, figures, ligand, md, mmpbsa, prepare


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[..., Any]

    def to_anthropic(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def _path(s) -> Path:
    return Path(s).expanduser().resolve()


# ----- Tool handlers ---------------------------------------------------------

def t_fetch_pdb(*, pdb_id: str, cache_dir: str | None = None) -> dict:
    res = fetch.fetch_pdb(pdb_id, cache_dir=_path(cache_dir) if cache_dir else None)
    return {"pdb_id": res.pdb_id, "path": str(res.path), "source": res.source}


def t_prepare_ligand(*, name: str, smiles: str, out_dir: str) -> dict:
    lg = ligand.load_ligand(name, smiles, _path(out_dir))
    return {"name": lg.name, "sdf_path": str(lg.sdf_path),
            "ff_xml": str(lg.ff_xml) if lg.ff_xml else None}


def t_fix_pdb(*, in_pdb: str, out_pdb: str, ph: float = 7.4) -> dict:
    r = prepare.fix_pdb(_path(in_pdb), _path(out_pdb), ph=ph)
    return {"pdb_path": str(r.pdb_path), "n_atoms": r.n_atoms,
            "n_residues": r.n_residues, "n_chains": r.n_chains}


def t_embed_membrane(*, fixed_pdb: str, out_pdb: str, lipid: str = "POPC",
                     padding_nm: float = 1.5, ionic_strength_M: float = 0.15) -> dict:
    r = prepare.add_membrane_and_solvate(
        _path(fixed_pdb), _path(out_pdb),
        lipid=lipid, padding_nm=padding_nm, ionic_strength_M=ionic_strength_M,
    )
    return {"pdb_path": str(r.pdb_path), "n_atoms": r.n_atoms,
            "n_residues": r.n_residues, "n_chains": r.n_chains}


def t_run_md(*, prepared_pdb: str, out_dir: str, name: str,
             production_ns: float = 100.0, n_replicas: int = 1,
             ligand_sdf: str | None = None, ligand_ff_xml: str | None = None,
             temperature_K: float = 310.0) -> dict:
    base = md.MDConfig(
        out_dir=_path(out_dir), name=name,
        temperature_K=temperature_K, production_ns=production_ns,
        ligand_sdf=_path(ligand_sdf) if ligand_sdf else None,
        ligand_ff_xml=_path(ligand_ff_xml) if ligand_ff_xml else None,
    )
    if n_replicas <= 1:
        r = md.run_md(_path(prepared_pdb), base)
        return {"replicas": [{"name": r.name, "topology": str(r.topology_path),
                              "trajectory": str(r.trajectory_path),
                              "log": str(r.log_path), "walltime_s": r.walltime_s,
                              "n_atoms": r.n_atoms}]}
    rs = md.run_replicas(_path(prepared_pdb), base, n_replicas)
    return {"replicas": [{"name": r.name, "topology": str(r.topology_path),
                          "trajectory": str(r.trajectory_path),
                          "log": str(r.log_path), "walltime_s": r.walltime_s,
                          "n_atoms": r.n_atoms} for r in rs]}


def t_run_mmpbsa(*, topology_pdb: str, trajectory_dcd: str, ligand_resname: str,
                 out_dir: str, ligand_ff_xml: str | None = None,
                 stride: int = 10, max_frames: int | None = None) -> dict:
    cfg = mmpbsa.MMPBSAConfig(
        topology_pdb=_path(topology_pdb), trajectory_dcd=_path(trajectory_dcd),
        ligand_resname=ligand_resname, out_dir=_path(out_dir),
        ligand_ff_xml=_path(ligand_ff_xml) if ligand_ff_xml else None,
        stride=stride, max_frames=max_frames,
    )
    r = mmpbsa.run_mmpbsa(cfg)
    return {"csv_path": str(r.csv_path), "json_path": str(r.json_path),
            "top_residues": r.top_residues, "mean_dG_total_kcal": r.mean_dG_total_kcal}


def t_analyze_trajectory(*, topology_pdb: str, trajectory_dcd: str, out_dir: str,
                         distances: list[dict] | None = None,
                         reference_pdb: str | None = None,
                         occupancy_cutoff_nm: float = 0.35) -> dict:
    specs = [analyze.DistanceSpec(**d) for d in (distances or [])]
    cfg = analyze.AnalysisConfig(
        topology_pdb=_path(topology_pdb), trajectory_dcd=_path(trajectory_dcd),
        out_dir=_path(out_dir),
        reference_pdb=_path(reference_pdb) if reference_pdb else None,
        distances=specs, occupancy_cutoff_nm=occupancy_cutoff_nm,
    )
    r = analyze.run_analysis(cfg)
    return {"rmsd_csv": str(r.rmsd_csv), "rmsf_csv": str(r.rmsf_csv),
            "distances_csv": str(r.distances_csv) if r.distances_csv else None,
            "occupancy_json": str(r.occupancy_json) if r.occupancy_json else None}


def t_make_figures(*, per_residue_csv: str, distances_csv: str | None,
                   rmsf_csv: str | None, out_dir: str) -> dict:
    out_dir_p = _path(out_dir)
    paths = {"per_residue_dG": str(figures.plot_per_residue_dG(
        _path(per_residue_csv), out_dir_p / "per_residue_dG.png"))}
    if distances_csv:
        paths["distances"] = str(figures.plot_distances(
            _path(distances_csv), out_dir_p / "distances.png"))
    if rmsf_csv:
        paths["rmsf"] = str(figures.plot_rmsf(
            _path(rmsf_csv), out_dir_p / "rmsf.png"))
    return {"figures": paths}


def t_write_report(*, expected_path: str, mmpbsa_summary_path: str,
                   occupancy_json: str | None, out_md: str) -> dict:
    import yaml

    expected = yaml.safe_load(_path(expected_path).read_text())
    summary = json.loads(_path(mmpbsa_summary_path).read_text())
    occupancy = json.loads(_path(occupancy_json).read_text()) if occupancy_json else None
    p = figures.write_report(_path(out_md), expected=expected,
                             mmpbsa_summary=summary, occupancy=occupancy)
    return {"report": str(p)}


# ----- Tool registry ---------------------------------------------------------

TOOLS: list[Tool] = [
    Tool("fetch_pdb",
         "Resolve a PDB id to a local file via cache → repo data → mirror → RCSB.",
         {"type": "object",
          "properties": {"pdb_id": {"type": "string"},
                         "cache_dir": {"type": "string"}},
          "required": ["pdb_id"]},
         t_fetch_pdb),
    Tool("prepare_ligand",
         "Build a 3D conformer from SMILES with RDKit and write an SDF.",
         {"type": "object",
          "properties": {"name": {"type": "string"},
                         "smiles": {"type": "string"},
                         "out_dir": {"type": "string"}},
          "required": ["name", "smiles", "out_dir"]},
         t_prepare_ligand),
    Tool("fix_pdb",
         "Repair a PDB with PDBFixer (missing atoms/residues, hydrogens at given pH).",
         {"type": "object",
          "properties": {"in_pdb": {"type": "string"},
                         "out_pdb": {"type": "string"},
                         "ph": {"type": "number"}},
          "required": ["in_pdb", "out_pdb"]},
         t_fix_pdb),
    Tool("embed_membrane_and_solvate",
         "Embed protein in POPC bilayer and solvate (TIP3P + NaCl) with OpenMM Modeller.",
         {"type": "object",
          "properties": {"fixed_pdb": {"type": "string"},
                         "out_pdb": {"type": "string"},
                         "lipid": {"type": "string"},
                         "padding_nm": {"type": "number"},
                         "ionic_strength_M": {"type": "number"}},
          "required": ["fixed_pdb", "out_pdb"]},
         t_embed_membrane),
    Tool("run_md",
         "Run minimize → NVT → NPT → production MD on OpenMM (CUDA preferred).",
         {"type": "object",
          "properties": {"prepared_pdb": {"type": "string"},
                         "out_dir": {"type": "string"},
                         "name": {"type": "string"},
                         "production_ns": {"type": "number"},
                         "n_replicas": {"type": "integer"},
                         "ligand_sdf": {"type": "string"},
                         "ligand_ff_xml": {"type": "string"},
                         "temperature_K": {"type": "number"}},
          "required": ["prepared_pdb", "out_dir", "name"]},
         t_run_md),
    Tool("run_mmpbsa",
         "Per-residue MM/GBSA decomposition over the trajectory.",
         {"type": "object",
          "properties": {"topology_pdb": {"type": "string"},
                         "trajectory_dcd": {"type": "string"},
                         "ligand_resname": {"type": "string"},
                         "out_dir": {"type": "string"},
                         "ligand_ff_xml": {"type": "string"},
                         "stride": {"type": "integer"},
                         "max_frames": {"type": "integer"}},
          "required": ["topology_pdb", "trajectory_dcd", "ligand_resname", "out_dir"]},
         t_run_mmpbsa),
    Tool("analyze_trajectory",
         "Compute RMSD/RMSF and tracked-distance time series + H-bond occupancies.",
         {"type": "object",
          "properties": {"topology_pdb": {"type": "string"},
                         "trajectory_dcd": {"type": "string"},
                         "out_dir": {"type": "string"},
                         "reference_pdb": {"type": "string"},
                         "occupancy_cutoff_nm": {"type": "number"},
                         "distances": {
                             "type": "array",
                             "items": {
                                 "type": "object",
                                 "properties": {
                                     "name": {"type": "string"},
                                     "residue_index": {"type": "integer"},
                                     "residue_atom": {"type": "string"},
                                     "ligand_resname": {"type": "string"},
                                     "ligand_atom": {"type": "string"},
                                 },
                                 "required": ["name", "residue_index", "residue_atom",
                                              "ligand_resname", "ligand_atom"],
                             }}},
          "required": ["topology_pdb", "trajectory_dcd", "out_dir"]},
         t_analyze_trajectory),
    Tool("make_figures",
         "Render per-residue ΔG bar chart, H-bond distance series, and RMSF plot.",
         {"type": "object",
          "properties": {"per_residue_csv": {"type": "string"},
                         "distances_csv": {"type": "string"},
                         "rmsf_csv": {"type": "string"},
                         "out_dir": {"type": "string"}},
          "required": ["per_residue_csv", "out_dir"]},
         t_make_figures),
    Tool("write_report",
         "Produce a side-by-side reproduction report against expected.yaml.",
         {"type": "object",
          "properties": {"expected_path": {"type": "string"},
                         "mmpbsa_summary_path": {"type": "string"},
                         "occupancy_json": {"type": "string"},
                         "out_md": {"type": "string"}},
          "required": ["expected_path", "mmpbsa_summary_path", "out_md"]},
         t_write_report),
]


def tool_by_name(name: str) -> Tool:
    for t in TOOLS:
        if t.name == name:
            return t
    raise KeyError(f"Unknown tool: {name}")


def anthropic_tool_schemas() -> list[dict]:
    return [t.to_anthropic() for t in TOOLS]
