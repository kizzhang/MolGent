"""Resolve PDB / EMDB / generic URL files via a fallback chain.

Order: explicit local cache -> repo data/ directory -> user-configured GitHub raw mirror
-> RCSB direct. The MolGent sandbox cannot reach RCSB; users stage files into
``data/structures/`` (committed) or point ``MOLGENT_PDB_MIRROR`` at a raw GitHub URL.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import requests

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = REPO_ROOT / "data" / "structures"
RCSB_TEMPLATE = "https://files.rcsb.org/download/{pdb}.pdb"


@dataclass
class FetchResult:
    pdb_id: str
    path: Path
    source: str


def _try_url(url: str, dest: Path, timeout: int = 20) -> bool:
    try:
        r = requests.get(url, timeout=timeout, stream=True)
    except requests.RequestException as exc:
        log.debug("GET %s failed: %s", url, exc)
        return False
    if r.status_code != 200:
        log.debug("GET %s -> %s", url, r.status_code)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 14):
            if chunk:
                f.write(chunk)
    return True


def fetch_pdb(pdb_id: str, cache_dir: Path | None = None) -> FetchResult:
    """Resolve ``pdb_id`` to a local .pdb path, trying sources in order.

    Sources, in order:
      1. ``cache_dir/<pdbid>.pdb`` if it already exists.
      2. ``MolGent/data/structures/<pdbid>.pdb`` (the repo-staged copy).
      3. ``$MOLGENT_PDB_MIRROR/<pdbid>.pdb`` — raw GitHub mirror configured by the user.
      4. RCSB ``files.rcsb.org`` (only works when network policy allows it).

    Raises FileNotFoundError if all sources miss.
    """

    pdb_id = pdb_id.lower()
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    cache_path = cache_dir / f"{pdb_id}.pdb"

    if cache_path.exists() and cache_path.stat().st_size > 0:
        return FetchResult(pdb_id, cache_path, "cache")

    repo_staged = DEFAULT_CACHE / f"{pdb_id}.pdb"
    if repo_staged.exists() and repo_staged.stat().st_size > 0:
        if repo_staged != cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo_staged, cache_path)
        return FetchResult(pdb_id, cache_path, "repo-staged")

    mirror = os.environ.get("MOLGENT_PDB_MIRROR")
    if mirror:
        url = mirror.rstrip("/") + f"/{pdb_id}.pdb"
        if _try_url(url, cache_path):
            return FetchResult(pdb_id, cache_path, f"mirror:{mirror}")

    if _try_url(RCSB_TEMPLATE.format(pdb=pdb_id), cache_path):
        return FetchResult(pdb_id, cache_path, "rcsb")

    raise FileNotFoundError(
        f"Could not resolve PDB {pdb_id!r}. Tried cache, repo-staged "
        f"({repo_staged}), mirror env MOLGENT_PDB_MIRROR, and RCSB."
    )


def stage_pdb(source: Path | str, pdb_id: str, cache_dir: Path | None = None) -> Path:
    """Copy a manually downloaded PDB into the cache. Useful on the sandbox."""

    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{pdb_id.lower()}.pdb"
    shutil.copy2(source, dest)
    return dest
