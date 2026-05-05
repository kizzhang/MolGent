"""Test scaffolding: dependency probes used as pytest skip-conditions."""

from __future__ import annotations

import importlib.util


def has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


HAS_RDKIT = has("rdkit")
HAS_OPENMM = has("openmm")
HAS_PDBFIXER = has("pdbfixer")
HAS_MDA = has("MDAnalysis")
HAS_MDTRAJ = has("mdtraj")
HAS_ANTHROPIC = has("anthropic")
HAS_YAML = has("yaml")
