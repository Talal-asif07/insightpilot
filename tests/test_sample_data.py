import pandas as pd

from data.generate_sample import INJECTION_NOTE, N_DUPLICATES, N_ORDERS, generate


def test_generator_is_deterministic():
    pd.testing.assert_frame_equal(generate(), generate())


def test_planted_issues_present():
    df = generate()
    assert len(df) == N_ORDERS + N_DUPLICATES
    assert df.duplicated().sum() == N_DUPLICATES
    assert df["revenue"].isna().any() and df["customer_rating"].isna().any()
    assert df["unit_price"].str.startswith("£").all()
    assert (df["region"] != df["region"].str.strip().str.title()).any()
    assert (df["notes"] == INJECTION_NOTE).sum() >= 1
    assert df["quantity"].max() >= 250


def test_committed_csv_matches_generator(sample_csv_path):
    committed = pd.read_csv(sample_csv_path, keep_default_na=False, na_values=[""])
    assert len(committed) == N_ORDERS + N_DUPLICATES
