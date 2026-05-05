"""Generic ``molgent`` CLI: run the agent against a workflow request."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent import AgentConfig, run_workflow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="molgent",
                                     description="Drive an MD reproduction workflow.")
    parser.add_argument("--prompt", "-p", help="Inline prompt for the agent")
    parser.add_argument("--prompt-file", "-f", type=Path,
                        help="File containing the workflow prompt")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--max-turns", type=int, default=30)
    args = parser.parse_args(argv)

    if args.prompt_file:
        text = args.prompt_file.read_text()
    elif args.prompt:
        text = args.prompt
    else:
        text = sys.stdin.read()

    cfg = AgentConfig(model=args.model, max_turns=args.max_turns)
    run = run_workflow(text, cfg)
    print(run.final_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
