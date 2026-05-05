"""Verify the fetch fallback chain without hitting the network."""

from __future__ import annotations

from pathlib import Path

import pytest

from molgent.core import fetch


def test_cache_hit(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "1abc.pdb").write_text("HEADER fake\nEND\n")
    res = fetch.fetch_pdb("1ABC", cache_dir=cache)
    assert res.source == "cache"
    assert res.path.exists()
    assert res.pdb_id == "1abc"


def test_repo_staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the file is missing from the user cache but lives in the repo
    `data/structures` directory, stage it transparently."""

    repo_data = tmp_path / "repo_data"
    repo_data.mkdir()
    (repo_data / "9zzz.pdb").write_text("HEADER repo-staged\nEND\n")

    monkeypatch.setattr(fetch, "DEFAULT_CACHE", repo_data)
    user_cache = tmp_path / "user_cache"
    res = fetch.fetch_pdb("9zzz", cache_dir=user_cache)

    assert res.source == "repo-staged"
    assert (user_cache / "9zzz.pdb").exists()


def test_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "DEFAULT_CACHE", tmp_path / "empty")
    monkeypatch.delenv("MOLGENT_PDB_MIRROR", raising=False)

    # Force the network attempt to fail without making a real request.
    def _no(*args, **kwargs):
        return False

    monkeypatch.setattr(fetch, "_try_url", _no)
    with pytest.raises(FileNotFoundError):
        fetch.fetch_pdb("0xxx", cache_dir=tmp_path / "missing_cache")


def test_stage_pdb(tmp_path: Path) -> None:
    src = tmp_path / "src.pdb"
    src.write_text("HEADER hand-staged\nEND\n")
    cache = tmp_path / "cache"
    out = fetch.stage_pdb(src, "7XJA", cache_dir=cache)
    assert out == cache / "7xja.pdb"
    assert out.read_text() == src.read_text()
