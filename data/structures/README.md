# Structure cache

The MolGent CI sandbox cannot reach `files.rcsb.org`. Stage the deposited
structures into this directory once on a network-permitted host so the rest
of the pipeline runs offline.

```
# On a host with internet access:
for id in 7XF5 7XJA 8GQU; do
    curl -L -o "$(echo $id | tr A-Z a-z).pdb" \
        "https://files.rcsb.org/download/${id}.pdb"
done
mv 7xf5.pdb 7xja.pdb 8gqu.pdb data/structures/
git add data/structures/*.pdb
git commit -m "stage ClC-2 structures (7XF5, 7XJA, 8GQU)"
```

Alternative: set `MOLGENT_PDB_MIRROR` to a raw GitHub URL whose layout is
`<base>/<pdbid>.pdb` (lowercase). `core/fetch.py` will pull from there before
attempting RCSB.

## Source

Deposited alongside Ma et al., *Nat. Commun.* **14**, 3424 (2023),
DOI 10.1038/s41467-023-39218-6.

| PDB | EMDB | State |
|---|---|---|
| 7XF5 | EMD-33169 | apo full-length ClC-2 |
| 7XJA | EMD-33223 | apo TMD |
| 8GQU | EMD-34202 | AK-42-bound TMD |
