"""Chart creation from a structured specification.

The model fills in a small spec (chart type, columns, aggregation) instead of writing free-form
plotting code. This makes charts consistent, labelled and much less error-prone. The aggregated
data behind each chart is returned to the model, so it can describe the chart accurately.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

CHART_TYPES = ("bar", "line", "scatter", "histogram", "box")
AGGREGATIONS = ("sum", "mean", "median", "count", "none")
TIME_GRAINS = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
MAX_CATEGORIES = 25
MAX_SERIES = 8
MAX_SCATTER_POINTS = 5000


class ChartError(Exception):
    """A chart spec that cannot be drawn; the message tells the model how to fix it."""


@dataclass
class Chart:
    title: str
    png: bytes
    data: pd.DataFrame
    spec: dict
    notes: list[str] = field(default_factory=list)


def _require_columns(df: pd.DataFrame, cols) -> None:
    missing = [c for c in cols if c and c not in df.columns]
    if missing:
        raise ChartError(f"Column(s) not found: {missing}. Available columns: {list(df.columns)}")


def _require_numeric(df: pd.DataFrame, col: str) -> None:
    if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
        raise ChartError(
            f"Column '{col}' is {df[col].dtype}, not numeric. Clean it first with run_python, e.g. "
            f"df['{col}'] = pd.to_numeric(df['{col}'].astype(str).str.replace(r'[£$,%]', '', regex=True), errors='coerce')"
        )


def _x_values(df: pd.DataFrame, x: str, grain: str | None) -> pd.Series:
    if grain and grain != "none":
        if grain not in TIME_GRAINS:
            raise ChartError(f"time_grain must be one of {list(TIME_GRAINS)} or 'none'.")
        if not pd.api.types.is_datetime64_any_dtype(df[x]):
            raise ChartError(f"time_grain needs '{x}' to be a date column, but it is {df[x].dtype}. "
                             f"Convert it first: df['{x}'] = pd.to_datetime(df['{x}'], dayfirst=True)")
        return df[x].dt.to_period(TIME_GRAINS[grain]).dt.to_timestamp()
    return df[x]


def _aggregate(df, x, y, agg, group, grain) -> tuple[pd.DataFrame, str]:
    if agg not in AGGREGATIONS:
        raise ChartError(f"aggregation must be one of {AGGREGATIONS}.")
    work = pd.DataFrame({x: _x_values(df, x, grain)})
    keys = [x]
    if group:
        work[group] = df[group].fillna("(missing)").astype(str)
        keys.append(group)
    if agg == "count":
        label = f"count of {y}" if y else "count"
        work[label] = df[y] if y else 1
        grouped = work.groupby(keys)[label].count() if y else work.groupby(keys)[label].size()
        return grouped.reset_index(name=label), label
    if not y:
        raise ChartError("y is required unless aggregation is 'count'.")
    _require_numeric(df, y)
    label = f"{agg} of {y}" if agg != "none" else y
    work[label] = df[y]
    if agg == "none":
        return work.dropna(), label
    return work.groupby(keys)[label].agg(agg).reset_index(), label


def _format_axis(ax, axis="y"):
    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}".rstrip("0").rstrip("."))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _date_labels(values: pd.Series, grain: str | None) -> list[str]:
    if not pd.api.types.is_datetime64_any_dtype(values):
        return [str(v) for v in values]
    if grain == "quarter":
        return [f"{d.year} Q{d.quarter}" for d in values]
    fmt = {"month": "%b %Y", "year": "%Y"}.get(grain or "", "%d %b %Y")
    return [d.strftime(fmt) for d in values]


def _top_series(data: pd.DataFrame, group: str, label: str) -> tuple[pd.DataFrame, list[str]]:
    totals = data.groupby(group)[label].sum().sort_values(ascending=False)
    if len(totals) <= MAX_SERIES:
        return data, []
    keep = totals.index[:MAX_SERIES]
    return data[data[group].isin(keep)], [f"Only the top {MAX_SERIES} of {len(totals)} '{group}' groups are shown."]


def build_chart(df: pd.DataFrame, spec: dict) -> Chart:
    ctype = spec.get("chart_type")
    if ctype not in CHART_TYPES:
        raise ChartError(f"chart_type must be one of {CHART_TYPES}.")
    x, y, group = spec.get("x"), spec.get("y"), spec.get("group_by")
    agg = spec.get("aggregation") or ("none" if ctype in ("scatter", "box", "histogram") else "sum")
    grain = spec.get("time_grain") or "none"
    top_n = spec.get("top_n")
    if not x and ctype != "box":
        raise ChartError("x is required.")
    _require_columns(df, [x, y, group])

    notes: list[str] = []
    fig = Figure(figsize=(9, 5), dpi=100)
    ax = fig.subplots()

    if ctype in ("bar", "line"):
        data, label = _aggregate(df, x, y, agg, group, grain)
        if agg == "none":
            raise ChartError("Bar and line charts need an aggregation (sum, mean, median or count).")
        is_time = pd.api.types.is_datetime64_any_dtype(data[x])
        if group:
            data, extra = _top_series(data, group, label)
            notes += extra
            pivot = data.pivot_table(index=x, columns=group, values=label, aggfunc="sum").sort_index()
            if ctype == "bar" and not is_time and len(pivot) > MAX_CATEGORIES:
                pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index[:MAX_CATEGORIES]]
                notes.append(f"Only the top {MAX_CATEGORIES} '{x}' values are shown.")
            pivot.index = _date_labels(pd.Series(pivot.index), grain) if is_time else pivot.index.astype(str)
            colors = PALETTE[: pivot.shape[1]]
            if ctype == "bar":
                pivot.plot(kind="bar", ax=ax, color=colors, width=0.8)
            else:
                pivot.plot(kind="line", ax=ax, color=colors, marker="o" if len(pivot) <= 30 else None, linewidth=2)
            ax.legend(title=group, fontsize=8, title_fontsize=9, frameon=False)
        else:
            if is_time or spec.get("sort") == "none":
                data = data.sort_values(x)
            else:
                data = data.sort_values(label, ascending=spec.get("sort") == "asc")
            if top_n:
                data = data.head(int(top_n))
            elif ctype == "bar" and len(data) > MAX_CATEGORIES and not is_time:
                notes.append(f"Only the top {MAX_CATEGORIES} of {len(data)} categories are shown.")
                data = data.head(MAX_CATEGORIES)
            labels = _date_labels(data[x], grain) if is_time else data[x].astype(str).tolist()
            if ctype == "bar":
                ax.bar(labels, data[label], color=PALETTE[0])
            else:
                ax.plot(labels, data[label], color=PALETTE[0], marker="o" if len(data) <= 30 else None, linewidth=2)
        ax.set_xlabel(x)
        ax.set_ylabel(label)
        _format_axis(ax)
        if ax.get_xticklabels() and len(ax.get_xticklabels()) > 6:
            for tick in ax.get_xticklabels():
                tick.set_rotation(45)
                tick.set_horizontalalignment("right")

    elif ctype == "scatter":
        if not y:
            raise ChartError("Scatter charts need both x and y.")
        _require_numeric(df, x)
        _require_numeric(df, y)
        cols = [x, y] + ([group] if group else [])
        pts = df[cols].dropna()
        if len(pts) > MAX_SCATTER_POINTS:
            notes.append(f"A random sample of {MAX_SCATTER_POINTS:,} of {len(pts):,} points is shown.")
            pts = pts.sample(MAX_SCATTER_POINTS, random_state=0)
        if group:
            for i, (g, part) in enumerate(list(pts.groupby(group))[:MAX_SERIES]):
                ax.scatter(part[x], part[y], s=14, alpha=0.6, color=PALETTE[i % len(PALETTE)], label=str(g))
            ax.legend(title=group, fontsize=8, frameon=False)
        else:
            ax.scatter(pts[x], pts[y], s=14, alpha=0.5, color=PALETTE[0])
        full = df[[x, y]].dropna()
        r = full[x].corr(full[y]) if len(full) > 2 else float("nan")
        data = pd.DataFrame({"n_points": [len(full)], "pearson_r": [round(float(r), 3)]})
        notes.append("Correlation describes association only, not causation.")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        _format_axis(ax, "x")
        _format_axis(ax, "y")

    elif ctype == "histogram":
        _require_numeric(df, x)
        values = df[x].dropna()
        bins = int(spec.get("bins") or 30)
        ax.hist(values, bins=bins, color=PALETTE[0], edgecolor="white")
        data = values.describe().round(2).to_frame("value").reset_index(names="statistic")
        ax.set_xlabel(x)
        ax.set_ylabel("frequency")
        _format_axis(ax, "x")

    else:  # box
        target = y or x
        _require_numeric(df, target)
        if y and x:
            top = df[x].value_counts().index[:MAX_CATEGORIES // 2]
            sub = df[df[x].isin(top)]
            groups = [(str(k), g[y].dropna()) for k, g in sub.groupby(x)]
            ax.boxplot([g for _, g in groups], tick_labels=[k for k, _ in groups], patch_artist=True,
                       boxprops={"facecolor": "#c9d6ea"}, medianprops={"color": PALETTE[1]})
            data = sub.groupby(x)[y].describe().round(2).reset_index()
            ax.set_xlabel(x)
            for tick in ax.get_xticklabels():
                tick.set_rotation(30)
                tick.set_horizontalalignment("right")
        else:
            ax.boxplot([df[target].dropna()], tick_labels=[target], patch_artist=True,
                       boxprops={"facecolor": "#c9d6ea"}, medianprops={"color": PALETTE[1]})
            data = df[target].describe().round(2).to_frame("value").reset_index(names="statistic")
        ax.set_ylabel(target)
        _format_axis(ax)

    title = spec.get("title") or f"{ctype.title()} chart of {y or x}"
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    return Chart(title=title, png=buffer.getvalue(), data=data.reset_index(drop=True), spec=dict(spec), notes=notes)
