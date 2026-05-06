# CHARMM-GUI Membrane Builder bundles

The paper-faithful pipeline (`molgent-clc2-gmx`) starts from CHARMM-GUI's
Membrane Builder output. CHARMM-GUI is interactive and cannot be scripted;
this directory holds the user-supplied bundles.

## Expected layout

```
data/charmm_gui/
  apo_tmd/                # 7XJA assembled with POPC + 150 mM NaCl
    charmm-gui-XXXXXXXX/
      gromacs/
        step5_input.gro
        topol.top
        index.ndx
        toppar/...
  bound_tmd/              # 8GQU + GH6 (AK-42) embedded
    charmm-gui-YYYYYYYY/
      gromacs/
        step5_input.gro
        topol.top
        index.ndx
        toppar/...
```

The wrapper auto-detects the inner ``charmm-gui-*/gromacs/`` subdirectory, so
you can also flatten one level if you prefer:

```
data/charmm_gui/bound_tmd/gromacs/step5_input.gro
                          /topol.top
                          /index.ndx
                          /toppar/
```

## Generation steps (one-time, manual)

1. Sign in at https://www.charmm-gui.org/.
2. **Membrane Builder → Bilayer Builder**.
3. Upload `data/structures/7xja.pdb` (apo) or `data/structures/8gqu.pdb`
   (bound). For the bound run, in the heterogen step **keep `GH6`** and check
   "Generate ligand topology — CGenFF" so CHARMM-GUI bundles the .str into the
   archive itself; otherwise upload your own `data/ligands/gh6.str`.
4. Paper conditions: POPC bilayer, 150 mM NaCl, water box, **CHARMM36m**
   force field, **GROMACS** output checkbox.
5. Download the resulting `charmm-gui-<id>.tgz` and extract here.

## Why is this manual?

CHARMM-GUI's terms of service prohibit scripting. The output is a fixed-input
artefact; once the bundle exists, everything downstream
(`molgent-clc2-gmx cpu-verify` / `run`) is fully reproducible.
