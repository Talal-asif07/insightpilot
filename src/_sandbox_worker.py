"""Runs one analysis code snippet in an isolated process. Called by sandbox.execute_code.

Usage: python _sandbox_worker.py <input.pkl> <output.pkl>
"""

import ast
import builtins
import contextlib
import io
import pickle
import sys
import traceback

import numpy as np
import pandas as pd

SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "format", "frozenset",
        "int", "isinstance", "len", "list", "map", "max", "min", "print", "range", "repr",
        "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
        "ValueError", "KeyError", "TypeError", "IndexError", "ZeroDivisionError", "Exception",
        "True", "False", "None",
    )
}


class _CappedStringIO(io.StringIO):
    def __init__(self, cap: int):
        super().__init__()
        self.cap = cap
        self.overflow = False

    def write(self, s):
        if self.tell() >= self.cap:
            self.overflow = True
            return len(s)
        return super().write(s)


def _format_value(value) -> str:
    with pd.option_context("display.max_rows", 60, "display.max_columns", 30, "display.width", 200):
        if isinstance(value, (pd.DataFrame, pd.Series)):
            return value.to_string(max_rows=60)
        if isinstance(value, np.generic):  # np.float64(3.5) -> 3.5
            value = value.item()
        elif isinstance(value, (list, tuple, dict)):
            return str(value)
        return repr(value) if isinstance(value, str) else str(value)


def _format_error(exc: BaseException) -> str:
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == "<analysis>"]
    where = f" (line {frames[-1].lineno})" if frames else ""
    return f"{type(exc).__name__}{where}: {exc}"


def main(in_path: str, out_path: str) -> None:
    with open(in_path, "rb") as fh:
        payload = pickle.load(fh)

    namespace = {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np, "df": payload["df"]}
    stdout = _CappedStringIO(payload.get("max_stdout", 20_000))
    result, error = None, None
    try:
        tree = ast.parse(payload["code"])
        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = ast.Expression(tree.body.pop().value)
        with contextlib.redirect_stdout(stdout):
            exec(compile(tree, "<analysis>", "exec"), namespace)
            if last_expr is not None:
                value = eval(compile(last_expr, "<analysis>", "eval"), namespace)
                if value is not None:
                    result = _format_value(value)
    except Exception as exc:  # report every user-code error back to the model
        error = _format_error(exc)

    text = stdout.getvalue()
    if stdout.overflow:
        text += "\n[print output truncated]"
    out = {"stdout": text, "result": result, "error": error, "df": namespace.get("df")}
    if not isinstance(out["df"], pd.DataFrame):
        out["df"] = None
        if error is None:
            out["error"] = "`df` must remain a pandas DataFrame. Assign results to a new variable instead."
    with open(out_path, "wb") as fh:
        pickle.dump(out, fh)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
