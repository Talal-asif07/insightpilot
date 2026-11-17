import pandas as pd

from src.analysis import format_profile, iqr_outliers, label_variants, numeric_text_ratio, profile_dataframe, quality_report
from src.charts import build_chart
from src.tools import TOOL_DEFINITIONS, ToolExecutor


# --- analysis ----------------------------------------------------------------------------
def test_numeric_text_detection(sales_df):
    assert numeric_text_ratio(sales_df["price_text"]) == 1.0
    assert numeric_text_ratio(sales_df["region"]) == 0.0
    assert numeric_text_ratio(sales_df["revenue"]) == 0.0  # already numeric


def test_label_variants(sales_df):
    variants = label_variants(sales_df["region"])
    assert set(variants) == {"london", "north"}


def test_iqr_outliers_detects_extreme_value():
    s = pd.Series(list(range(1, 50)) + [10_000])
    out = iqr_outliers(s)
    assert out["count"] == 1 and out["max"] == 10_000


def test_profile_flags(sales_df):
    profile = profile_dataframe(sales_df)
    flags = {c["name"]: c["flags"] for c in profile["columns"]}
    assert any("stored as text" in f for f in flags["price_text"])
    assert any("inconsistent" in f for f in flags["region"])
    text = format_profile(sales_df, profile)
    assert "Rows: 20" in text and "First 5 rows" in text


def test_quality_report_finds_planted_issues(sales_df):
    issues = quality_report(sales_df)
    found = {(i["column"], i["issue"]) for i in issues}
    assert ("(all)", "Exact duplicate rows") in found
    assert ("region", "Inconsistent category labels") in found
    assert ("price_text", "Numbers stored as text") in found
    assert ("revenue", "Missing values") in found
    assert issues[0]["severity"] == "high"


# --- charts ------------------------------------------------------------------------------
def test_bar_chart_aggregates(sales_df):
    chart = build_chart(sales_df, {"chart_type": "bar", "x": "category", "y": "revenue", "aggregation": "sum", "title": "t"})
    assert chart.png.startswith(b"\x89PNG")
    totals = dict(zip(chart.data["category"], chart.data["sum of revenue"]))
    assert totals["B"] == 2602.0


def test_line_chart_with_month_grain(sales_df):
    chart = build_chart(sales_df, {"chart_type": "line", "x": "order_date", "y": "revenue",
                                   "time_grain": "month", "title": "trend"})
    assert len(chart.data) == 3


def test_count_chart_without_y(sales_df):
    chart = build_chart(sales_df, {"chart_type": "bar", "x": "category", "aggregation": "count", "title": "n"})
    assert chart.data["count"].sum() == 20


# --- tool executor -----------------------------------------------------------------------
def test_tool_schemas_are_well_formed():
    names = {t["name"] for t in TOOL_DEFINITIONS}
    assert names == {"profile_dataset", "run_python", "create_chart", "data_quality_report"}
    for tool in TOOL_DEFINITIONS:
        assert tool["input_schema"]["type"] == "object" and tool["description"]


def test_unknown_tool_is_an_error(sales_df):
    out = ToolExecutor(sales_df).execute("delete_everything", {})
    assert out.is_error and "Unknown tool" in out.text


def test_chart_errors_are_helpful(sales_df):
    ex = ToolExecutor(sales_df)
    out = ex.execute("create_chart", {"chart_type": "bar", "x": "nope", "y": "revenue", "title": "t"})
    assert out.is_error and "Available columns" in out.text
    out = ex.execute("create_chart", {"chart_type": "bar", "x": "region", "y": "price_text", "title": "t"})
    assert out.is_error and "not numeric" in out.text and "pd.to_numeric" in out.text


def test_run_python_cleaning_persists_and_is_logged(sales_df):
    ex = ToolExecutor(sales_df)
    out = ex.execute("run_python", {"code": "df = df.drop_duplicates()", "purpose": "Removed duplicate rows"})
    assert not out.is_error and "Dataset updated" in out.text
    assert len(ex.df) == 6 and ex.change_log == ["Removed duplicate rows"]
    assert "Rows: 6" in ex.dataset_context()  # profile cache invalidated
    ex.reset()
    assert len(ex.df) == 20 and not ex.is_modified


def test_run_python_error_does_not_change_data(sales_df):
    ex = ToolExecutor(sales_df)
    out = ex.execute("run_python", {"code": "df = df.head(1)\nraise ValueError('boom')", "purpose": "x"})
    assert out.is_error and len(ex.df) == 20
