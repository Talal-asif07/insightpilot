"""Terminal interface: python -m src.cli data/sample_retail.csv"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import config
from .agent import AgentAPIError, DataAnalystAgent, create_client
from .data_loader import DataLoadError, load_file
from .tools import TOOL_LABELS, ToolExecutor


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with InsightPilot in the terminal.")
    parser.add_argument("file", help="CSV or Excel file to analyse")
    parser.add_argument("--model", default=config.DEFAULT_MODEL)
    parser.add_argument("--charts-dir", default="outputs", help="Where to save charts")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    path = Path(args.file)
    try:
        loaded = load_file(path.read_bytes(), path.name)
        client = create_client(os.getenv("ANTHROPIC_API_KEY"))
    except (DataLoadError, AgentAPIError, OSError) as exc:
        raise SystemExit(f"Error: {exc}") from None

    print(f"Loaded {path.name}: {len(loaded.df):,} rows x {loaded.df.shape[1]} columns")
    for note in loaded.notes:
        print(f"  note: {note}")

    agent = DataAnalystAgent(
        client, ToolExecutor(loaded.df), model=args.model,
        on_event=lambda kind, p: print(f"  ... {TOOL_LABELS.get(p['name'], p['name'])}")
        if kind == "tool_start" else None,
    )
    charts_dir = Path(args.charts_dir)
    print("Ask a question about your data (type 'exit' to quit).\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        response = agent.ask(question)
        print(f"\nInsightPilot: {response.text}\n")
        for chart in response.charts:
            charts_dir.mkdir(exist_ok=True)
            out = charts_dir / f"{len(list(charts_dir.glob('*.png'))) + 1:02d}_chart.png"
            out.write_bytes(chart.png)
            print(f"  [chart saved: {out}]")
        print(f"  ({len(response.steps)} steps, {response.elapsed_seconds}s, "
              f"{response.usage.get('input_tokens', 0):,} in / {response.usage.get('output_tokens', 0):,} out tokens)\n")


if __name__ == "__main__":
    main()
