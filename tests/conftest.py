import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def sales_df() -> pd.DataFrame:
    return pd.DataFrame({
        "order_date": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-03", "2024-02-15", "2024-03-01"] * 4),
        "region": ["London", "london", "North", "North ", "South"] * 4,
        "category": ["A", "B", "A", "C", "B"] * 4,
        "price_text": ["£1,200.00", "£15.50", "£3.00", "£40.00", "£5.25"] * 4,
        "revenue": [100.0, 250.5, 80.0, None, 400.0] * 4,
        "quantity": [1, 2, 1, 3, 2] * 3 + [1, 2, 1, 900, 2],
    })


@pytest.fixture
def sample_csv_path() -> Path:
    return ROOT / "data" / "sample_retail.csv"
