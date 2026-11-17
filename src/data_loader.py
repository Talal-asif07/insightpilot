"""Robust file loading: encodings, delimiters, Excel sheets, column names, dates and size limits.

Real-world files are messy. Every automatic fix made here is recorded in ``LoadResult.notes``
so it can be shown to the user. The loader never changes the data silently.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field

import pandas as pd

from . import config


class DataLoadError(Exception):
    """Raised with a user-friendly message when a file cannot be loaded."""


@dataclass
class LoadResult:
    df: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    column_mapping: dict[str, str] = field(default_factory=dict)  # clean name -> original name
    sheet_names: list[str] = field(default_factory=list)
    sampled: bool = False


ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
DELIMITERS = (",", ";", "\t", "|")
EXCEL_EXTENSIONS = {"xlsx", "xlsm", "xls"}
TEXT_EXTENSIONS = {"csv", "tsv", "txt"}

_DATE_SHAPE = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}")
UK_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d/%m/%y")
US_FORMATS = ("%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y")
DATE_PARSE_THRESHOLD = 0.95


# --- Text decoding & delimiter detection -------------------------------------------------
def decode_bytes(raw: bytes) -> tuple[str, str]:
    """Decode bytes, trying UTF-8 first and then common Windows/legacy encodings."""
    for encoding in ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise DataLoadError("Could not read the file's text encoding. Save it as UTF-8 CSV and try again.")


def detect_delimiter(text: str) -> str:
    """Choose the delimiter that gives a consistent column count (>1) over the first lines.

    This is more reliable than ``csv.Sniffer`` on files with quoted values such as "£1,234.50".
    """
    lines = [line for line in text.splitlines()[:50] if line.strip()]
    if not lines:
        return ","
    best, best_score = ",", (False, 0)
    for delim in DELIMITERS:
        counts = [len(row) for row in csv.reader(lines, delimiter=delim)]
        mode = max(set(counts), key=counts.count)
        if mode < 2:
            continue
        consistent = counts.count(mode) / len(counts) >= 0.9
        score = (consistent, mode)
        if score > best_score:
            best, best_score = delim, score
    return best


# --- Column names ------------------------------------------------------------------------
def normalise_columns(columns) -> tuple[list[str], dict[str, str]]:
    """Convert column names to snake_case identifiers that are safe in code; keep a mapping."""
    clean_names: list[str] = []
    mapping: dict[str, str] = {}
    for i, col in enumerate(columns):
        name = re.sub(r"[^0-9a-z]+", "_", str(col).strip().lower()).strip("_")
        name = name or f"column_{i + 1}"
        if name[0].isdigit():
            name = f"col_{name}"
        base, k = name, 2
        while name in mapping:
            name = f"{base}_{k}"
            k += 1
        clean_names.append(name)
        mapping[name] = str(col)
    return clean_names, mapping


# --- Dates -------------------------------------------------------------------------------
def _is_text(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _best_rate(values: pd.Series, formats) -> tuple[str | None, float]:
    best_fmt, best_rate = None, 0.0
    for fmt in formats:
        rate = pd.to_datetime(values, format=fmt, errors="coerce").notna().mean()
        if rate > best_rate:
            best_fmt, best_rate = fmt, float(rate)
    return best_fmt, best_rate


def parse_date_columns(df: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    """Convert text columns that clearly hold dates. Ambiguous DD/MM vs MM/DD defaults to UK."""
    for col in df.columns:
        series = df[col]
        if not _is_text(series):
            continue
        values = series.dropna().astype(str).str.strip()
        values = values[values != ""]
        if values.empty or values.head(100).str.match(_DATE_SHAPE).mean() < 0.8:
            continue

        iso = pd.to_datetime(values, format="ISO8601", errors="coerce")
        if iso.notna().mean() >= DATE_PARSE_THRESHOLD:
            df[col] = pd.to_datetime(series, format="ISO8601", errors="coerce")
            notes.append(f"Column '{col}' converted to dates (ISO format).")
            continue

        uk_fmt, uk_rate = _best_rate(values, UK_FORMATS)
        us_fmt, us_rate = _best_rate(values, US_FORMATS)
        if uk_rate >= DATE_PARSE_THRESHOLD and us_rate >= DATE_PARSE_THRESHOLD:
            fmt = uk_fmt
            notes.append(
                f"Column '{col}' converted to dates. The format is ambiguous (every day value is 12 or "
                "less), so UK day-first order (DD/MM/YYYY) was assumed. Check this if your data is from the US."
            )
        elif uk_rate >= DATE_PARSE_THRESHOLD:
            fmt = uk_fmt
            notes.append(f"Column '{col}' converted to dates (day-first, {uk_fmt}).")
        elif us_rate >= DATE_PARSE_THRESHOLD:
            fmt = us_fmt
            notes.append(f"Column '{col}' converted to dates (month-first, {us_fmt}).")
        else:
            continue

        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        failed = int(parsed.isna().sum() - series.isna().sum())
        if failed > 0:
            notes.append(f"{failed:,} value(s) in '{col}' could not be read as dates and were set to missing.")
        df[col] = parsed
    return df


# --- Main entry point --------------------------------------------------------------------
def _read_excel(raw: bytes, sheet_name: str | None) -> tuple[pd.DataFrame, list[str], list[str]]:
    notes: list[str] = []
    try:
        workbook = pd.ExcelFile(io.BytesIO(raw))
    except ImportError as exc:
        raise DataLoadError(f"Excel support is missing a package: {exc}. Install openpyxl (and xlrd for .xls).") from exc
    except Exception as exc:  # corrupted or password-protected files
        raise DataLoadError(f"Could not open the Excel file ({type(exc).__name__}). Is it corrupted or password protected?") from exc
    sheets = [str(s) for s in workbook.sheet_names]
    sheet = sheet_name or sheets[0]
    if sheet not in sheets:
        raise DataLoadError(f"Sheet '{sheet}' not found. Available sheets: {', '.join(sheets)}")
    df = workbook.parse(sheet)
    if len(sheets) > 1:
        notes.append(f"The workbook has {len(sheets)} sheets; loaded '{sheet}'.")
    return df, notes, sheets


def _read_text(raw: bytes, extension: str) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    text, encoding = decode_bytes(raw)
    if encoding != "utf-8-sig":
        notes.append(f"The file is not UTF-8; it was read as {encoding}.")
    delimiter = "\t" if extension == "tsv" else detect_delimiter(text)
    if delimiter != ",":
        label = {"\t": "tab", ";": "semicolon", "|": "pipe"}[delimiter]
        notes.append(f"Detected a {label}-separated file.")
    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError("The file contains no data.") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"The file could not be parsed as a table: {exc}") from exc
    return df, notes


def load_file(raw: bytes, filename: str, sheet_name: str | None = None) -> LoadResult:
    """Load an uploaded CSV/TSV/Excel file into a cleaned-up DataFrame with notes."""
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > config.MAX_FILE_MB:
        raise DataLoadError(f"The file is {size_mb:.1f} MB; the limit is {config.MAX_FILE_MB} MB.")
    if not raw or not raw.strip():
        raise DataLoadError("The file is empty.")

    extension = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    sheets: list[str] = []
    if extension in EXCEL_EXTENSIONS:
        df, notes, sheets = _read_excel(raw, sheet_name)
    elif extension in TEXT_EXTENSIONS or extension == "":
        df, notes = _read_text(raw, extension)
    else:
        raise DataLoadError(f"Unsupported file type '.{extension}'. Upload a CSV, TSV or Excel file.")

    result = _finalise(df, notes)
    result.sheet_names = sheets
    return result


def _finalise(df: pd.DataFrame, notes: list[str]) -> LoadResult:
    rows_before, cols_before = df.shape
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.shape[1] < cols_before:
        notes.append(f"Removed {cols_before - df.shape[1]} completely empty column(s).")
    if df.shape[0] < rows_before:
        notes.append(f"Removed {rows_before - df.shape[0]} completely empty row(s).")
    if df.empty or df.shape[1] == 0:
        raise DataLoadError("The file has headers but no data rows.")

    clean, mapping = normalise_columns(df.columns)
    if any(c != mapping[c] for c in clean):
        notes.append("Column names were converted to snake_case so they work in code (original names are kept).")
    df.columns = clean
    df = df.reset_index(drop=True)

    sampled = False
    if len(df) > config.MAX_ROWS:
        notes.append(
            f"The file has {len(df):,} rows; a random sample of {config.MAX_ROWS:,} rows is used. "
            "Totals and counts describe the sample, not the full file."
        )
        df = df.sample(config.MAX_ROWS, random_state=42).sort_index().reset_index(drop=True)
        sampled = True

    df = parse_date_columns(df, notes)
    return LoadResult(df=df, notes=notes, column_mapping=mapping, sampled=sampled)
