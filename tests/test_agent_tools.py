"""Schema + dispatch sanity checks for the agent tool layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .conftest import HAS_ANTHROPIC


def test_tools_module_importable() -> None:
    # Importing the tools module must not require anthropic; only the agent
    # loop does.
    from molgent.agent import tools

    schemas = tools.anthropic_tool_schemas()
    names = [s["name"] for s in schemas]
    assert "fetch_pdb" in names
    assert "run_md" in names
    assert "run_mmpbsa" in names
    assert "write_report" in names
    for s in schemas:
        assert set(s.keys()) >= {"name", "description", "input_schema"}
        assert s["input_schema"]["type"] == "object"


def test_tool_dispatch_matches_handler() -> None:
    from molgent.agent import tools

    for t in tools.TOOLS:
        # Every declared required field must appear in the handler signature
        import inspect

        sig = inspect.signature(t.handler)
        for req in t.input_schema.get("required", []):
            assert req in sig.parameters, f"{t.name} missing handler arg {req}"


def test_fetch_dispatch(tmp_path: Path) -> None:
    from molgent.agent import tools
    from molgent.core import fetch

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "1abc.pdb").write_text("HEADER fake\nEND\n")
    out = tools.t_fetch_pdb(pdb_id="1ABC", cache_dir=str(cache))
    assert out["pdb_id"] == "1abc"
    assert out["source"] == "cache"
    assert Path(out["path"]).exists()


@pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic SDK not installed")
def test_agent_module_importable() -> None:
    from molgent.agent import agent  # noqa: F401
