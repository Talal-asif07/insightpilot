"""Guarded execution of model-written pandas code.

There are two layers of defence:

1. Static checks (AST) reject imports, file and network access, private/dunder attribute access
   and dangerous built-ins before anything runs.
2. The code runs in a separate Python process with restricted built-ins and a hard timeout,
   so infinite loops and crashes cannot freeze or break the app.

This is NOT a hardened security sandbox. For a public, multi-user production service, run
the code in an isolated container (for example Docker or a managed sandbox such as E2B).
See the "Security" section of the README.
"""

from __future__ import annotations

import ast
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config

WORKER_PATH = Path(__file__).with_name("_sandbox_worker.py")

BLOCKED_NAMES = {
    "open", "eval", "exec", "compile", "__import__", "globals", "locals", "vars", "input",
    "breakpoint", "exit", "quit", "help", "getattr", "setattr", "delattr", "memoryview",
    "__builtins__", "__loader__", "__spec__",
}
BLOCKED_ATTRIBUTES = {
    # filesystem / OS access reachable through libraries
    "os", "sys", "io", "subprocess", "shutil", "pathlib", "importlib", "ctypes", "builtins",
    "system", "popen", "spawn",
    # pandas / numpy file I/O
    "to_csv", "to_excel", "to_pickle", "to_parquet", "to_feather", "to_hdf", "to_sql",
    "to_stata", "to_html", "to_json", "to_latex", "to_xml", "to_clipboard", "to_orc",
    "load", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt",
    "fromfile", "tofile", "memmap", "DataSource",
}


class SecurityViolation(Exception):
    """Raised when code fails the static safety checks."""


@dataclass
class ExecutionResult:
    ok: bool
    stdout: str = ""
    result: str | None = None
    error: str | None = None
    df: pd.DataFrame | None = None
    df_changed: bool = False


def validate_code(code: str) -> None:
    """Raise SecurityViolation if the code uses anything outside the allowed analysis subset."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SecurityViolation(f"Syntax error on line {exc.lineno}: {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SecurityViolation("Imports are not allowed. `pd`, `np` and `df` are already available.")
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise SecurityViolation(f"Use of `{node.id}` is not allowed.")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise SecurityViolation(f"Access to private attribute `{node.attr}` is not allowed.")
            if node.attr in BLOCKED_ATTRIBUTES or node.attr.startswith("read_"):
                raise SecurityViolation(f"`.{node.attr}` is not allowed (file, system or network access).")
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise SecurityViolation("`global` and `nonlocal` are not allowed.")


def execute_code(code: str, df: pd.DataFrame, timeout: int | None = None) -> ExecutionResult:
    """Validate, then run the code in a separate process against a copy of ``df``."""
    timeout = timeout or config.CODE_TIMEOUT_SECONDS
    try:
        validate_code(code)
    except SecurityViolation as exc:
        return ExecutionResult(ok=False, error=f"Blocked by safety checks: {exc}")

    with tempfile.TemporaryDirectory(prefix="insightpilot_") as tmp:
        in_path, out_path = Path(tmp) / "in.pkl", Path(tmp) / "out.pkl"
        with open(in_path, "wb") as fh:
            pickle.dump({"code": code, "df": df, "max_stdout": config.MAX_STDOUT_CHARS}, fh)
        try:
            proc = subprocess.run(
                [sys.executable, str(WORKER_PATH), str(in_path), str(out_path)],
                cwd=tmp, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                ok=False,
                error=f"Execution stopped: the code took longer than {timeout} seconds. "
                "Use a simpler or vectorised approach (avoid row-by-row loops).",
            )
        if not out_path.exists():
            detail = (proc.stderr or "").strip().splitlines()[-1:] or ["no output"]
            return ExecutionResult(ok=False, error=f"The execution process crashed: {detail[0]}")
        with open(out_path, "rb") as fh:
            out = pickle.load(fh)

    new_df = out.get("df")
    changed = isinstance(new_df, pd.DataFrame) and out.get("error") is None and not _same_frame(df, new_df)
    return ExecutionResult(
        ok=out.get("error") is None,
        stdout=out.get("stdout", ""),
        result=out.get("result"),
        error=out.get("error"),
        df=new_df if changed else None,
        df_changed=changed,
    )


def _same_frame(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    if a.shape != b.shape or list(a.columns) != list(b.columns) or list(a.dtypes) != list(b.dtypes):
        return False
    try:
        return bool(a.equals(b))
    except Exception:
        return False
