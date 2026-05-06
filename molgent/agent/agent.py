"""Agent loop: drive a workflow with Claude + the MolGent tool registry.

Uses the Anthropic SDK with prompt caching: the system prompt and tool block
are marked ``cache_control`` so subsequent turns within a workflow reuse the
cached prefix.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

from .prompts import SYSTEM_PROMPT
from .tools import anthropic_tool_schemas, tool_by_name

log = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    max_turns: int = 30
    api_key_env: str = "ANTHROPIC_API_KEY"


@dataclass
class AgentRun:
    final_text: str
    transcript: list[dict] = field(default_factory=list)


def _client(cfg: AgentConfig):
    import anthropic

    api_key = os.environ.get(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{cfg.api_key_env} not set; required to drive the MolGent agent."
        )
    return anthropic.Anthropic(api_key=api_key)


def _system_block() -> list[dict]:
    return [{"type": "text", "text": SYSTEM_PROMPT,
             "cache_control": {"type": "ephemeral"}}]


def _tools_block() -> list[dict]:
    schemas = anthropic_tool_schemas()
    # Mark the last tool with cache_control to cache the whole tool block.
    if schemas:
        schemas[-1] = {**schemas[-1], "cache_control": {"type": "ephemeral"}}
    return schemas


def _dispatch(tool_use: dict) -> dict:
    name = tool_use["name"]
    args = tool_use.get("input", {})
    log.info("tool_call %s %s", name, args)
    try:
        result = tool_by_name(name).handler(**args)
    except Exception as exc:  # noqa: BLE001
        log.exception("tool %s failed", name)
        return {"is_error": True, "content": f"{type(exc).__name__}: {exc}"}
    return {"is_error": False, "content": json.dumps(result, default=str)}


def run_workflow(user_message: str, cfg: AgentConfig | None = None) -> AgentRun:
    """Drive a single workflow request with the agent until it stops."""

    cfg = cfg or AgentConfig()
    client = _client(cfg)
    messages: list[dict] = [{"role": "user", "content": user_message}]
    transcript: list[dict] = []

    for turn in range(cfg.max_turns):
        resp = client.messages.create(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            system=_system_block(),
            tools=_tools_block(),
            messages=messages,
        )
        transcript.append({"turn": turn, "stop_reason": resp.stop_reason,
                           "usage": resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage)})

        # Convert response content blocks into a serialisable assistant message
        assistant_blocks = [b.model_dump() if hasattr(b, "model_dump") else dict(b)
                            for b in resp.content]
        messages.append({"role": "assistant", "content": assistant_blocks})

        if resp.stop_reason == "end_turn":
            text = "\n".join(b["text"] for b in assistant_blocks if b.get("type") == "text")
            return AgentRun(final_text=text, transcript=transcript)

        if resp.stop_reason == "tool_use":
            tool_results = []
            for block in assistant_blocks:
                if block.get("type") == "tool_use":
                    out = _dispatch(block)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "is_error": out["is_error"],
                        "content": out["content"],
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason
        text = "\n".join(b.get("text", "") for b in assistant_blocks if b.get("type") == "text")
        return AgentRun(final_text=text or f"stopped: {resp.stop_reason}", transcript=transcript)

    return AgentRun(final_text="max_turns reached", transcript=transcript)
