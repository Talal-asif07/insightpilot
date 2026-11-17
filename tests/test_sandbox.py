import pandas as pd
import pytest

from src.sandbox import SecurityViolation, execute_code, validate_code

BLOCKED = [
    "import os",
    "from pathlib import Path",
    "open('/etc/passwd').read()",
    "eval('1+1')",
    "__import__('os')",
    "df.__class__.__bases__",
    "().__class__.__subclasses__()",
    "getattr(df, 'to_csv')('x.csv')",
    "df.to_csv('stolen.csv')",
    "pd.read_csv('/etc/passwd')",
    "np.save('x.npy', [1])",
    "pd.io.common",
    "globals()",
]


@pytest.mark.parametrize("code", BLOCKED)
def test_dangerous_code_is_blocked(code):
    with pytest.raises(SecurityViolation):
        validate_code(code)


def test_blocked_code_returns_error_result():
    res = execute_code("import os", pd.DataFrame({"a": [1]}))
    assert not res.ok and "Imports are not allowed" in res.error


def test_syntax_error_reported_with_line():
    with pytest.raises(SecurityViolation, match="line 2"):
        validate_code("x = 1\nx = (")


def test_last_expression_is_returned(sales_df):
    res = execute_code("total = df['revenue'].sum()\nround(total, 2)", sales_df)
    assert res.ok and res.result == "3322.0"


def test_print_output_captured(sales_df):
    res = execute_code("print('rows', len(df))", sales_df)
    assert res.ok and res.stdout.strip() == "rows 20"


def test_dataframe_modification_is_returned(sales_df):
    res = execute_code("df = df[df['region'] == 'London']", sales_df)
    assert res.ok and res.df_changed and len(res.df) == 4
    assert len(sales_df) == 20  # the original is never mutated in place


def test_read_only_code_does_not_change_df(sales_df):
    res = execute_code("df.groupby('category')['revenue'].mean()", sales_df)
    assert res.ok and not res.df_changed and res.df is None


def test_runtime_error_has_type_and_line(sales_df):
    res = execute_code("x = 1\ndf['missing_column']", sales_df)
    assert not res.ok and res.error.startswith("KeyError (line 2)")


def test_infinite_loop_times_out(sales_df):
    res = execute_code("while True:\n    pass", sales_df, timeout=2)
    assert not res.ok and "longer than 2 seconds" in res.error


def test_df_must_stay_a_dataframe(sales_df):
    res = execute_code("df = df['revenue'].sum()", sales_df)
    assert not res.ok and "must remain a pandas DataFrame" in res.error


def test_variables_do_not_persist(sales_df):
    assert execute_code("my_var = 5", sales_df).ok
    res = execute_code("my_var", sales_df)
    assert not res.ok and "NameError" in res.error
