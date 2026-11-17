"""Run the golden evaluation set against the real model and write a results report.

Usage:
    python -m evals.run_evals                       # all cases
    python -m evals.run_evals --cases rows dupes    # selected cases
    python -m evals.run_evals --model claude-haiku-5-5 --judge-model claude-haiku-5-5

Needs ANTHROPIC_API_KEY. Results are saved to evals/results/ as JSON and Markdown.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from src import config
from src.agent import DataAnalystAgent, create_client
from src.data_loader import load_file
from src.tools import ToolExecutor

from .golden_set import CASES, DATA_PATH, Case, load_truth_frame

RESULTS_DIR = Path(__file__).with_name("results")
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")

JUDGE_PROMPT = """You are grading an AI data analyst's answer. Apply the rubric strictly.

Question: {question}

Rubric: {rubric}

Answer being graded:
<answer>
{answer}
</answer>

Analysis steps the agent ran (for context):
<steps>
{steps}
</steps>

Reply with exactly one line: "PASS: <short reason>" or "FAIL: <short reason>"."""


def extract_numbers(text: str) -> list[float]:
    values = []
    for match in NUMBER.findall(text.replace("£", " ").replace("%", " ")):
        try:
            values.append(float(match.replace(",", "")))
        except ValueError:
            continue
    return values


def grade_numeric(case: Case, answer: str, truth_df) -> tuple[bool, str]:
    truth = case.truth(truth_df)
    targets = truth if isinstance(truth, list) else [truth]
    found = extract_numbers(answer)

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= max(case.abs_tolerance, abs(b) * case.tolerance, 1e-9)

    hit = any(close(v, t) for v in found for t in targets)
    mentions_ok = all(m.lower() in answer.lower() for m in case.must_mention)
    expected = ", ".join(f"{t:,.4g}" if abs(t) < 1000 else f"{t:,.2f}" for t in targets)
    reason = f"expected {expected}" + (f" and mention of {case.must_mention}" if case.must_mention else "")
    return hit and mentions_ok, reason


def grade_behaviour(client, judge_model: str, case: Case, answer: str, steps: str) -> tuple[bool, str]:
    response = client.messages.create(
        model=judge_model, max_tokens=200,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=case.question, rubric=case.rubric, answer=answer, steps=steps[:6000])}],
    )
    verdict = "".join(b.text for b in response.content if b.type == "text").strip()
    return verdict.upper().startswith("PASS"), verdict


def run(case_ids: list[str] | None, model: str, judge_model: str) -> dict:
    client = create_client(os.getenv("ANTHROPIC_API_KEY"))
    loaded = load_file(DATA_PATH.read_bytes(), DATA_PATH.name)
    truth_df = load_truth_frame()
    cases = [c for c in CASES if not case_ids or c.id in case_ids]
    rows = []

    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.id}: {case.question}")
        agent = DataAnalystAgent(client, ToolExecutor(loaded.df), model=model)  # fresh state per case
        started = time.perf_counter()
        response = agent.ask(case.question)
        latency = round(time.perf_counter() - started, 1)
        steps_text = "\n\n".join(f"[{s.tool}] {json.dumps(s.input)[:500]}\n-> {s.output[:800]}" for s in response.steps)

        if response.stop_reason == "api_error":
            passed, reason = False, response.text
        elif case.kind == "numeric":
            passed, reason = grade_numeric(case, response.text, truth_df)
        else:
            passed, reason = grade_behaviour(client, judge_model, case, response.text, steps_text)

        tool_errors = sum(s.is_error for s in response.steps)
        rows.append({
            "id": case.id, "category": case.category, "kind": case.kind, "question": case.question,
            "passed": passed, "reason": reason, "answer": response.text, "stop_reason": response.stop_reason,
            "tool_calls": len(response.steps), "tool_errors": tool_errors,
            "recovered": tool_errors > 0 and response.stop_reason == "complete",
            "latency_s": latency, "input_tokens": response.usage.get("input_tokens", 0),
            "output_tokens": response.usage.get("output_tokens", 0),
        })
        print(f"    {'PASS' if passed else 'FAIL'} - {reason[:120]} ({len(response.steps)} tools, {latency}s)")

    n = len(rows)
    summary = {
        "model": model, "judge_model": judge_model, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cases": n,
        "passed": sum(r["passed"] for r in rows),
        "accuracy_pct": round(100 * sum(r["passed"] for r in rows) / n, 1) if n else 0,
        "avg_tool_calls": round(sum(r["tool_calls"] for r in rows) / n, 2) if n else 0,
        "tool_error_rate_pct": round(100 * sum(r["tool_errors"] for r in rows) / max(1, sum(r["tool_calls"] for r in rows)), 1),
        "self_corrected_cases": sum(r["recovered"] for r in rows),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / n, 1) if n else 0,
        "avg_input_tokens": round(sum(r["input_tokens"] for r in rows) / n) if n else 0,
        "avg_output_tokens": round(sum(r["output_tokens"] for r in rows) / n) if n else 0,
    }
    return {"summary": summary, "results": rows}


def write_report(report: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (RESULTS_DIR / f"results_{stamp}.json").write_text(json.dumps(report, indent=2))
    s = report["summary"]
    by_cat: dict[str, list[bool]] = {}
    for r in report["results"]:
        by_cat.setdefault(r["category"], []).append(r["passed"])
    lines = [
        f"# Evaluation results ({s['timestamp']})", "",
        f"Model: `{s['model']}` · Judge: `{s['judge_model']}`", "",
        "| Metric | Value |", "|---|---|",
        f"| Accuracy | **{s['passed']}/{s['cases']} ({s['accuracy_pct']}%)** |",
        f"| Avg tool calls per question | {s['avg_tool_calls']} |",
        f"| Tool error rate | {s['tool_error_rate_pct']}% |",
        f"| Cases that self-corrected after a tool error | {s['self_corrected_cases']} |",
        f"| Avg latency | {s['avg_latency_s']} s |",
        f"| Avg tokens (in / out) | {s['avg_input_tokens']:,} / {s['avg_output_tokens']:,} |",
        "", "## By category", "", "| Category | Passed |", "|---|---|",
        *[f"| {c} | {sum(v)}/{len(v)} |" for c, v in sorted(by_cat.items())],
        "", "## Cases", "", "| Case | Result | Tools | Reason |", "|---|---|---|---|",
        *[f"| {r['id']} | {'✅' if r['passed'] else '❌'} | {r['tool_calls']} | {r['reason'][:100].replace('|', '/')} |"
          for r in report["results"]],
    ]
    path = RESULTS_DIR / f"results_{stamp}.md"
    path.write_text("\n".join(lines))
    (RESULTS_DIR / "latest.md").write_text("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="*", help="Case ids to run (default: all)")
    parser.add_argument("--model", default=config.DEFAULT_MODEL)
    parser.add_argument("--judge-model", default=config.DEFAULT_MODEL)
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    report = run(args.cases, args.model, args.judge_model)
    path = write_report(report)
    s = report["summary"]
    print(f"\nAccuracy: {s['passed']}/{s['cases']} ({s['accuracy_pct']}%). Report: {path}")


if __name__ == "__main__":
    main()
