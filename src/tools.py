"""Tool definitions (the JSON schemas the model sees) and their execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from . import config
from .analysis import format_profile, format_quality_report, quality_report
from .charts import AGGREGATIONS, CHART_TYPES, TIME_GRAINS, Chart, ChartError, build_chart
from .sandbox import execute_code

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "profile_dataset",
        "description": (
            "Return the current dataset's shape, column types, missing values, unique counts, summary "
            "statistics, automatic quality flags and the first rows. Call it again after you modify `df`."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_python",
        "description": (
            "Execute pandas code against the dataset. `df` (the current dataset), `pd` and `np` are preloaded. "
            "Imports, file/network access and private attributes are blocked. The value of the last line is "
            "returned (like a notebook cell) and print() output is captured. Reassigning `df` "
            "(e.g. `df = df.drop_duplicates()`) permanently updates the dataset for later steps, so only do this "
            "for cleaning you will report to the user. Other variables are NOT kept between calls. Return "
            "aggregated results rather than whole tables: output longer than ~3,000 characters is truncated. "
            f"Time limit: {config.CODE_TIMEOUT_SECONDS} seconds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python/pandas code to execute."},
                "purpose": {"type": "string", "description": "One short sentence, shown to the user, saying what this step does."},
            },
            "required": ["code", "purpose"],
        },
    },
    {
        "name": "create_chart",
        "description": (
            "Draw a chart for the user and get back the data behind it. Choose the chart to fit the question: "
            "line for trends over time (with time_grain), bar for comparing categories, histogram for a "
            "distribution, box for spread/outliers by group, scatter for the relationship between two numeric "
            "columns. Columns must already be clean (numeric/date) - fix them with run_python first if needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chart_type": {"type": "string", "enum": list(CHART_TYPES)},
                "x": {"type": "string", "description": "Column for the x-axis (category or date). For histogram: the numeric column."},
                "y": {"type": "string", "description": "Numeric column to aggregate/plot. Optional when aggregation is 'count'."},
                "aggregation": {"type": "string", "enum": list(AGGREGATIONS), "description": "How to combine y per x value. Default: sum."},
                "group_by": {"type": "string", "description": "Optional column to split into separate series (max 8 shown)."},
                "time_grain": {"type": "string", "enum": ["none", *TIME_GRAINS], "description": "Bucket a date x-axis."},
                "top_n": {"type": "integer", "description": "Bar charts: keep only the top N categories."},
                "sort": {"type": "string", "enum": ["desc", "asc", "none"], "description": "Bar ordering. Default desc."},
                "bins": {"type": "integer", "description": "Histogram bins (default 30)."},
                "title": {"type": "string", "description": "Clear, specific chart title."},
            },
            "required": ["chart_type", "title"],
        },
    },
    {
        "name": "data_quality_report",
        "description": (
            "Run automated data-quality checks on the current dataset: duplicates, missing values, numbers "
            "stored as text, inconsistent category labels, IQR outliers, negative amounts and constant columns. "
            "Each issue comes with a severity and a suggested fix."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]

TOOL_LABELS = {
    "profile_dataset": "Profiling the dataset",
    "run_python": "Running analysis code",
    "create_chart": "Creating a chart",
    "data_quality_report": "Checking data quality",
}


@dataclass
class ToolOutput:
    text: str
    is_error: bool = False
    chart: Chart | None = None
    code: str | None = None


class ToolExecutor:
    """Holds the session's dataset and executes tool calls against it. Never raises."""

    def __init__(self, df: pd.DataFrame, timeout: int | None = None):
        self.original_df = df.copy()
        self.df = df.copy()
        self.timeout = timeout or config.CODE_TIMEOUT_SECONDS
        self.change_log: list[str] = []
        self._profile_cache: str | None = None

    # --- dataset state -------------------------------------------------------------------
    def dataset_context(self) -> str:
        if self._profile_cache is None:
            self._profile_cache = format_profile(self.df)
        return self._profile_cache

    def set_df(self, df: pd.DataFrame, reason: str) -> None:
        self.df = df
        self.change_log.append(reason)
        self._profile_cache = None

    def reset(self) -> None:
        self.df = self.original_df.copy()
        self.change_log.clear()
        self._profile_cache = None

    @property
    def is_modified(self) -> bool:
        return bool(self.change_log)

    # --- dispatch ------------------------------------------------------------------------
    def execute(self, name: str, tool_input: dict | None) -> ToolOutput:
        handlers = {
            "profile_dataset": self._profile,
            "run_python": self._run_python,
            "create_chart": self._create_chart,
            "data_quality_report": self._quality,
        }
        if name not in handlers:
            return ToolOutput(f"Unknown tool '{name}'. Available tools: {list(handlers)}", is_error=True)
        try:
            return handlers[name](tool_input or {})
        except Exception as exc:  # a tool bug must never crash the conversation
            return ToolOutput(f"Tool '{name}' failed unexpectedly: {type(exc).__name__}: {exc}", is_error=True)

    def _profile(self, _: dict) -> ToolOutput:
        return ToolOutput(self.dataset_context())

    def _quality(self, _: dict) -> ToolOutput:
        return ToolOutput(format_quality_report(quality_report(self.df)))

    def _run_python(self, tool_input: dict) -> ToolOutput:
        code = (tool_input.get("code") or "").strip()
        if not code:
            return ToolOutput("No code was provided.", is_error=True)
        result = execute_code(code, self.df, timeout=self.timeout)
        parts = []
        if result.stdout.strip():
            parts.append("stdout:\n" + result.stdout.rstrip())
        if result.result is not None:
            parts.append("result:\n" + result.result)
        if result.error:
            parts.append("error:\n" + result.error)
            return ToolOutput("\n\n".join(parts), is_error=True, code=code)
        if result.df_changed and result.df is not None:
            before = self.df.shape
            purpose = tool_input.get("purpose") or "Dataset modified"
            self.set_df(result.df, purpose)
            parts.append(f"Dataset updated: {before[0]:,} rows x {before[1]} cols -> "
                         f"{result.df.shape[0]:,} rows x {result.df.shape[1]} cols.")
        if not parts:
            parts.append("Code ran successfully with no output. End with an expression or use print() to see results.")
        return ToolOutput("\n\n".join(parts), code=code)

    def _create_chart(self, tool_input: dict) -> ToolOutput:
        try:
            chart = build_chart(self.df, tool_input)
        except ChartError as exc:
            return ToolOutput(f"Chart not created: {exc}", is_error=True)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            table = chart.data.to_string(index=False, max_rows=40)
        text = f"Chart '{chart.title}' is now displayed to the user.\nData behind the chart:\n{table}"
        if chart.notes:
            text += "\nNotes: " + " ".join(chart.notes)
        return ToolOutput(text, chart=chart)
