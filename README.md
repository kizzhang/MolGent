# MolGent

Agent-orchestrated MD + MM/PBSA reproduction workflows for structural-biology
papers. The first concrete workflow reproduces the per-residue binding
analysis of **Ma et al., *Nat. Commun.* 14, 3424 (2023) — "Cryo-EM structures
of ClC-2 chloride channel reveal the blocking mechanism of its specific
inhibitor AK-42"** (DOI [10.1038/s41467-023-39218-6](https://doi.org/10.1038/s41467-023-39218-6)).

## What it does

```
fetch_pdb ──► fix_pdb ──► embed_membrane_and_solvate ──► run_md ──► run_mmpbsa
                                                              │
                                                              └──► analyze_trajectory
                                                                          │
                                                          make_figures + write_report
```

Two ways to drive the pipeline:

1. **Deterministic CLI** (`molgent-clc2 run`) — runs the steps in fixed order
   from a YAML config. Use this for CI and reproducibility.
2. **Agent CLI** (`molgent --prompt-file workflow.md`) — feeds a paper-style
   prompt to Claude Sonnet 4.6 and lets the model decide tool order, retries,
   and report wording. Uses prompt caching for the system prompt + tool block.

## Install

The headline scientific deps install via pip:

```bash
pip install -e .
```

The ligand force-field path needs `openmmforcefields` (which transitively
needs `openff-toolkit` + `ambertools`); those are conda-only:

```bash
conda install -c conda-forge openmmforcefields openff-toolkit ambertools
pip install -e .
```

If you're stuck on pip-only, pre-generate the AK-42 OpenMM ForceField XML on a
conda host once and commit it as `data/ligands/AK4.xml`; the MD module picks it
up automatically.

## Reproduce Ma et al. 2023

1. **Stage the deposited structures** (see [`data/structures/README.md`](data/structures/README.md)).
   The sandbox blocks RCSB; commit the three `.pdb` files once.
2. **Run the deterministic pipeline**:
   ```bash
   molgent-clc2 run --config molgent/workflows/clc2_ak42/config.yaml
   ```
   This runs `100 ns × 3 replicas` for both the apo and AK-42-bound TMD on
   GPU. Wall time is ~2–4 days on a single recent NVIDIA GPU.
3. **Inspect the report**:
   ```
   data/results/clc2_ak42/report.md
   data/results/clc2_ak42/figures/{per_residue_dG,distances,rmsf}.png
   ```

### Acceptance criterion

The reproduction is considered to pass when:

- Top-4 residues by absolute MM/GBSA ΔG include ≥ 3 of the paper's reported
  set **{K204, S392, Q393, K394}**.
- The diagnostic H-bonds K204(NZ)↔AK-42 pyridine N and K394(N)↔AK-42 carboxyl
  O stay within 3.5 Å for ≥ 70% of the production trajectory.

The exact thresholds live in
[`molgent/workflows/clc2_ak42/expected.yaml`](molgent/workflows/clc2_ak42/expected.yaml)
and are checked by `core/figures.write_report`.

## Project layout

```
molgent/
  agent/        # Anthropic-SDK driver + tool registry
  core/         # fetch / ligand / prepare / md / mmpbsa / analyze / figures
  workflows/
    clc2_ak42/  # the paper-reproduction workflow
data/
  structures/   # staged PDBs (committed)
  ligands/      # AK-42 SDF (regenerated from SMILES) + optional FF xml
  results/      # MD trajectories + MM/PBSA outputs (gitignored)
tests/          # unit + smoke
```

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

Smoke MD test runs in <1 minute on CPU when openmm + pdbfixer are installed.
Other tests are pure Python and skip dependency-gated cases automatically.

## Status

| Component | State |
|---|---|
| Tool layer (`molgent/agent/tools.py`) | implemented + tested |
| Deterministic pipeline (`molgent-clc2`) | implemented |
| Agent loop with prompt caching | implemented |
| ClC-2/AK-42 workflow config + expected outcomes | committed |
| End-to-end reproduction run on GPU | **runs on the user's box** |

## License

MIT.
