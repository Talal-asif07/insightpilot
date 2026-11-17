import io

import pandas as pd
import pytest

from src import config
from src.data_loader import DataLoadError, detect_delimiter, load_file, normalise_columns


def test_loads_utf8_csv_and_normalises_columns():
    raw = b"Order ID,Total Sales (GBP),2024 Target\n1,10.5,20\n2,11.0,25\n"
    result = load_file(raw, "x.csv")
    assert list(result.df.columns) == ["order_id", "total_sales_gbp", "col_2024_target"]
    assert result.column_mapping["total_sales_gbp"] == "Total Sales (GBP)"
    assert result.df["total_sales_gbp"].sum() == pytest.approx(21.5)


def test_windows_encoding_is_detected():
    raw = "item,price\nTea,£2.50\n".encode("cp1252")
    result = load_file(raw, "legacy.csv")
    assert result.df.loc[0, "price"] == "£2.50"
    assert any("cp1252" in n for n in result.notes)


def test_semicolon_delimiter_with_quoted_commas():
    raw = b'name;amount;city\n"Smith, J";"1,200";Leeds\n"Khan, A";"300";York\n'
    result = load_file(raw, "euro.csv")
    assert result.df.shape == (2, 3)
    assert result.df.loc[0, "name"] == "Smith, J"
    assert any("semicolon" in n for n in result.notes)


def test_detect_delimiter_prefers_consistent_split():
    text = 'a,b,c\n"x,1",2,3\n"y,2",4,5\n'
    assert detect_delimiter(text) == ","
    assert detect_delimiter("a\tb\n1\t2\n") == "\t"


@pytest.mark.parametrize("raw", [b"", b"   \n  "])
def test_empty_file_raises(raw):
    with pytest.raises(DataLoadError, match="empty"):
        load_file(raw, "empty.csv")


def test_header_only_file_raises():
    with pytest.raises(DataLoadError, match="no data rows"):
        load_file(b"a,b,c\n", "headers.csv")


def test_unsupported_extension():
    with pytest.raises(DataLoadError, match="Unsupported"):
        load_file(b"data", "report.pdf")


def test_file_size_limit(monkeypatch):
    monkeypatch.setattr(config, "MAX_FILE_MB", 0)
    with pytest.raises(DataLoadError, match="limit"):
        load_file(b"a,b\n1,2\n", "big.csv")


def test_large_file_is_sampled(monkeypatch):
    monkeypatch.setattr(config, "MAX_ROWS", 10)
    raw = ("v\n" + "\n".join(str(i) for i in range(100))).encode()
    result = load_file(raw, "many.csv")
    assert len(result.df) == 10 and result.sampled
    assert any("sample" in n for n in result.notes)


def test_uk_dates_parsed_day_first():
    raw = b"date,value\n25/12/2024,1\n01/02/2024,2\n13/03/2024,3\n"
    df = load_file(raw, "d.csv").df
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df.loc[1, "date"] == pd.Timestamp("2024-02-01")


def test_ambiguous_dates_assume_uk_and_warn():
    raw = b"date,value\n01/02/2024,1\n03/04/2024,2\n05/06/2024,3\n"
    result = load_file(raw, "d.csv")
    assert result.df.loc[0, "date"] == pd.Timestamp("2024-02-01")
    assert any("ambiguous" in n for n in result.notes)


def test_us_dates_detected():
    raw = b"date,value\n12/25/2024,1\n02/13/2024,2\n"
    df = load_file(raw, "d.csv").df
    assert df.loc[0, "date"] == pd.Timestamp("2024-12-25")


def test_iso_dates_and_non_dates_untouched():
    raw = b"when,code\n2024-01-05,A1\n2024-02-06,B2\n"
    df = load_file(raw, "d.csv").df
    assert pd.api.types.is_datetime64_any_dtype(df["when"])
    assert not pd.api.types.is_datetime64_any_dtype(df["code"])


def test_empty_rows_and_columns_removed():
    raw = b"a,b,empty\n1,2,\n,,\n3,4,\n"
    result = load_file(raw, "x.csv")
    assert result.df.shape == (2, 2)


def test_duplicate_column_names_made_unique():
    names, _ = normalise_columns(["Sales", "sales", "SALES "])
    assert names == ["sales", "sales_2", "sales_3"]


def test_excel_with_multiple_sheets():
    pytest.importorskip("openpyxl")
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="First", index=False)
        pd.DataFrame({"b": [3, 4, 5]}).to_excel(writer, sheet_name="Second", index=False)
    raw = buffer.getvalue()
    first = load_file(raw, "book.xlsx")
    assert first.sheet_names == ["First", "Second"] and list(first.df.columns) == ["a"]
    second = load_file(raw, "book.xlsx", sheet_name="Second")
    assert len(second.df) == 3
    with pytest.raises(DataLoadError, match="not found"):
        load_file(raw, "book.xlsx", sheet_name="Missing")
