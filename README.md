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

1. **Paper-faithful CLI** (`molgent-clc2-gmx`) — GROMACS + CHARMM36m + CGenFF
   + CHARMM-GUI Membrane Builder, exactly the stack used in the paper. Output
   matches Ma et al. 2023's force field, thermostat (V-rescale), barostat
   (Parrinello-Rahman), and 5 ns × 5 replicates production schedule. Use this
   when you need the rankings to align with the paper. Requires a CHARMM-GUI
   account (manual web step) and a CGenFF-licensed station, plus a system
   GROMACS install.
2. **Reference CLI** (`molgent-clc2 run`) — OpenMM + AMBER14SB/LIPID17 + GAFF2
   reference implementation. Self-contained, single-package install, but the
   force field differs from the paper so absolute energies and occupancies are
   indicative rather than reproducible.
3. **Agent CLI** (`molgent --prompt-file workflow.md`) — feeds a paper-style
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

## Reproduce Ma et al. 2023 (paper-faithful pipeline)

The default reproduction path uses GROMACS + CHARMM36m. There is a one-time
manual prep before the CLI takes over.

### One-time prep (done off-box)

1. **CHARMM-GUI Membrane Builder** — open https://www.charmm-gui.org/ ,
   upload **7XJA** (apo TMD) and **8GQU** (bound TMD) separately, configure
   POPC bilayer + 150 mM NaCl, request GROMACS output. For the 8GQU run check
   "include heterogen" so the deposited GH6 ligand is preserved. Download both
   archives and extract them into:
   ```
   data/charmm_gui/apo_tmd/
   data/charmm_gui/bound_tmd/
   ```
2. **CGenFF for AK-42** — extract GH6 with `molgent-clc2-gmx`'s ligand helper
   or run the included one-liner once:
   ```bash
   python -c "from molgent.core.ligand import extract_from_pdb; \
              extract_from_pdb('data/structures/8gqu.pdb', 'GH6', 'A', \
              'OC(=O)c1cccnc1Nc2c(Cl)ccc(OCc3ccccc3)c2Cl', 'data/ligands')"
   ```
   Upload `data/ligands/GH6_from_pdb.mol2` to https://cgenff.silcsbio.com/
   and save the resulting stream file as `data/ligands/gh6.str`. (Alternative:
   if your CHARMM-GUI run already merged the ligand, the .str gets generated
   inside the bundle automatically.)
3. **GROMACS** — `brew install gromacs` (macOS) or your distro equivalent.
4. **gmx_MMPBSA** — `mamba install -c conda-forge ambertools=23 parmed=4`,
   then `pip install gmx_MMPBSA` in that env.

### Running the pipeline

* **CPU verification** (≈30 min on a recent laptop, single replicate ~ 50 ps
  each phase) — confirms the bundle parses, distances resolve, and gmx_MMPBSA
  consumes the trajectory:
  ```bash
  molgent-clc2-gmx cpu-verify
  ```
* **Full paper protocol** (5 ns × 5 replicates × {apo, bound}; needs GPU):
  ```bash
  molgent-clc2-gmx run
  ```
* **Inspect the report**:
  ```
  data/results/clc2_ak42_gmx/report.md
  data/results/clc2_ak42_gmx/figures/{per_residue_dG,distances,rmsf}.png
  ```

### Reference (OpenMM/AMBER) pipeline

The original OpenMM/AMBER reference path is still available for comparison and
for environments where GROMACS / AmberTools cannot be installed:

```bash
molgent-clc2 run --config molgent/workflows/clc2_ak42/config.yaml
```

Wall time is comparable on GPU. **Force field differs from the paper** so the
top-residue ranking and binding energy scale should be interpreted as
indicative.

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
| Reference pipeline (`molgent-clc2`, OpenMM/AMBER) | implemented |
| Paper-faithful pipeline (`molgent-clc2-gmx`, GROMACS/CHARMM36m) | implemented; needs CHARMM-GUI bundle + CGenFF .str + GROMACS install |
| Agent loop with prompt caching | implemented |
| ClC-2/AK-42 workflow config + expected outcomes | committed |
| End-to-end reproduction run on GPU | **runs on the user's box** |

## License

MIT.
