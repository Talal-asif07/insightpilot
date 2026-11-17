"""Dataset profiling and data-quality checks. These run without any LLM (deterministic)."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from . import config

_NUMERIC_JUNK = re.compile(r"[£$€¥,%\s]")
_AMOUNT_WORDS = ("price", "revenue", "amount", "sales", "cost", "quantity", "qty", "total")


def _is_text(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def numeric_text_ratio(series: pd.Series) -> float:
    """Share of non-empty text values that become numbers once £, $, %, commas and spaces are removed."""
    if not _is_text(series):
        return 0.0
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return 0.0
    cleaned = values.str.replace(_NUMERIC_JUNK, "", regex=True)
    return float(pd.to_numeric(cleaned, errors="coerce").notna().mean())


def label_variants(series: pd.Series, max_unique: int = 100) -> dict[str, list[str]]:
    """Find category labels that differ only by case or surrounding spaces, e.g. 'London' vs 'london '."""
    if not _is_text(series):
        return {}
    values = series.dropna().astype(str)
    values = values[values.str.strip() != ""]
    if values.empty or values.nunique() > max_unique:
        return {}
    frame = pd.DataFrame({"raw": values, "key": values.str.strip().str.lower()})
    groups = frame.groupby("key")["raw"].unique()
    return {key: sorted(repr(v) for v in raw) for key, raw in groups.items() if len(raw) > 1}


def iqr_outliers(series: pd.Series) -> dict[str, float] | None:
    """Tukey's rule: values beyond 1.5 x IQR from the quartiles."""
    if not _is_numeric(series):
        return None
    values = series.dropna()
    if values.nunique() < 10:
        return None
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return None
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (values < low) | (values > high)
    return {"count": int(mask.sum()), "lower": float(low), "upper": float(high),
            "min": float(values.min()), "max": float(values.max())}


def _looks_like_id(name: str, series: pd.Series) -> bool:
    non_null = series.dropna()
    return (name.endswith("id") or name.endswith("_id")) and len(non_null) > 0 and non_null.nunique() == len(non_null)


# --- Profile -----------------------------------------------------------------------------
def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    columns = []
    for name in df.columns:
        s = df[name]
        null_pct = float(s.isna().mean() * 100)
        info: dict[str, Any] = {
            "name": name,
            "dtype": str(s.dtype),
            "non_null": int(s.notna().sum()),
            "null_pct": round(null_pct, 1),
            "unique": int(s.nunique(dropna=True)),
            "flags": [],
        }
        if pd.api.types.is_datetime64_any_dtype(s) and s.notna().any():
            info["stats"] = {"min": str(s.min().date()), "max": str(s.max().date())}
        elif _is_numeric(s) and s.notna().any():
            info["stats"] = {k: round(float(v), 2) for k, v in
                             {"min": s.min(), "median": s.median(), "mean": s.mean(), "max": s.max()}.items()}
        samples = s.dropna().astype(str).unique()[:3]
        info["samples"] = [v if len(v) <= 40 else v[:37] + "..." for v in samples]

        if _looks_like_id(name, s):
            info["flags"].append("likely identifier")
        if numeric_text_ratio(s) >= 0.9:
            info["flags"].append("numbers stored as text (needs cleaning before maths)")
        if null_pct >= 20:
            info["flags"].append(f"high missing rate ({null_pct:.0f}%)")
        if info["unique"] == 1:
            info["flags"].append("constant (single value)")
        if label_variants(s):
            info["flags"].append("inconsistent labels (case/spacing)")
        columns.append(info)

    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells_pct": round(float(df.isna().mean().mean() * 100), 2) if df.size else 0.0,
        "columns": columns,
    }


def format_profile(df: pd.DataFrame, profile: dict[str, Any] | None = None) -> str:
    """Compact text profile for the model: schema, statistics, flags and a few sample rows."""
    profile = profile or profile_dataframe(df)
    lines = [
        f"Rows: {profile['n_rows']:,} | Columns: {profile['n_cols']} | "
        f"Exact duplicate rows: {profile['duplicate_rows']:,} | Missing cells: {profile['missing_cells_pct']}%",
        "",
        "Columns:",
    ]
    for c in profile["columns"]:
        line = f"- {c['name']} ({c['dtype']}): {c['null_pct']}% missing, {c['unique']:,} unique"
        if "stats" in c:
            line += " | " + ", ".join(f"{k}={v}" for k, v in c["stats"].items())
        line += f" | e.g. {c['samples']}"
        if c["flags"]:
            line += f" | FLAGS: {'; '.join(c['flags'])}"
        lines.append(line)
    with pd.option_context("display.max_columns", 50, "display.width", 250, "display.max_colwidth", 40):
        lines += ["", f"First {config.SAMPLE_ROWS_FOR_MODEL} rows:", df.head(config.SAMPLE_ROWS_FOR_MODEL).to_string()]
    return "\n".join(lines)


# --- Quality report ----------------------------------------------------------------------
def quality_report(df: pd.DataFrame) -> list[dict[str, str]]:
    """Return a list of issues, ordered by severity, each with a recommended fix."""
    issues: list[dict[str, str]] = []

    def add(severity, column, issue, detail, suggestion):
        issues.append({"severity": severity, "column": column, "issue": issue, "detail": detail, "suggestion": suggestion})

    dupes = int(df.duplicated().sum())
    if dupes:
        add("high", "(all)", "Exact duplicate rows", f"{dupes:,} rows ({dupes / len(df):.1%}) are exact copies of another row.",
            "Confirm they are not genuine repeat orders, then remove them with df.drop_duplicates().")

    for name in df.columns:
        s = df[name]
        missing = float(s.isna().mean())
        if missing >= 0.9:
            add("low", name, "Mostly empty column", f"{missing:.1%} of values are missing.",
                "Usually fine for optional free-text fields; do not use it for statistics.")
        elif missing > 0:
            sev = "high" if missing >= 0.2 else "medium" if missing >= 0.05 else "low"
            add(sev, name, "Missing values", f"{int(s.isna().sum()):,} missing ({missing:.1%}).",
                "Exclude them from calculations and say so, or impute only if the method is justified.")
        if numeric_text_ratio(s) >= 0.9:
            add("medium", name, "Numbers stored as text", f"Values such as {s.dropna().astype(str).iloc[0]!r} contain symbols or commas.",
                "Strip £/$/%/commas and convert with pd.to_numeric before any maths.")
        variants = label_variants(s)
        if variants:
            examples = "; ".join(", ".join(v) for v in list(variants.values())[:3])
            add("medium", name, "Inconsistent category labels",
                f"{len(variants)} label(s) appear in several spellings, e.g. {examples}.",
                "Standardise with .str.strip() and consistent capitalisation before grouping.")
        if _looks_like_id(name, s):
            continue
        out = iqr_outliers(s)
        if out and out["count"]:
            add("low" if out["count"] / max(len(df), 1) < 0.05 else "medium", name, "Potential outliers (IQR rule)",
                f"{out['count']:,} values outside [{out['lower']:,.2f}, {out['upper']:,.2f}]; max is {out['max']:,.2f}.",
                "Investigate before removing; report the median alongside the mean.")
        if _is_numeric(s) and any(w in name for w in _AMOUNT_WORDS) and (s < 0).any():
            add("medium", name, "Negative values", f"{int((s < 0).sum()):,} negative values in an amount-like column.",
                "Check whether they are refunds or errors.")
        if s.nunique(dropna=True) == 1:
            add("low", name, "Constant column", "Only one distinct value.", "It adds no information to the analysis.")

    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(issues, key=lambda i: order[i["severity"]])


def format_quality_report(issues: list[dict[str, str]]) -> str:
    if not issues:
        return "No data-quality issues detected by the automated checks."
    lines = [f"{len(issues)} potential issue(s) found:"]
    for i in issues:
        lines.append(f"- [{i['severity'].upper()}] {i['column']}: {i['issue']}. {i['detail']} Suggested fix: {i['suggestion']}")
    return "\n".join(lines)
