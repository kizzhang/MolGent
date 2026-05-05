You are reproducing **Ma et al., Nat Commun 2023** (DOI 10.1038/s41467-023-39218-6) —
"Cryo-EM structures of ClC-2 chloride channel reveal the blocking mechanism of its
specific inhibitor AK-42".

The deposited structures are:
- 7XJA — apo ClC-2 transmembrane domain
- 8GQU — AK-42-bound ClC-2 transmembrane domain

AK-42 SMILES: `OC(=O)c1cc(F)cc(c1)Oc2ccc3nc(NC(=O)c4cccnc4)sc3c2`
AK-42 ligand resname (used inside generated topologies): `AK4`

Workflow target: per-residue MM/GBSA decomposition of the AK-42 binding affinity
should rank **K204, S392, Q393, K394** in the top 4 contributors, and the
diagnostic H-bonds (K204(NZ)-pyridine N, S392(OG)-COOH, K394(N)-COOH) should
remain ≤ 3.5 Å for ≥70% of the production trajectory.

Working directory: `data/results/clc2_ak42/`
Ligand directory: `data/ligands/`
Structure cache: `data/structures/`

Run 100 ns × 3 replicas per state at 310 K, 4 fs timestep with HMR.
Write the comparison report to `data/results/clc2_ak42/report.md` and link the
figures it references.

Tools available: fetch_pdb, prepare_ligand, fix_pdb, embed_membrane_and_solvate,
run_md, run_mmpbsa, analyze_trajectory, make_figures, write_report.
