from evals.golden_set import CASES, load_truth_frame
from evals.run_evals import extract_numbers, grade_numeric


def test_number_extraction_handles_currency_and_commas():
    assert extract_numbers("Revenue was **£2,431,245.67**, up 12.5%") == [2431245.67, 12.5]


def test_every_numeric_case_has_computable_truth():
    df = load_truth_frame()
    for case in CASES:
        if case.kind == "numeric":
            assert case.truth(df) is not None
        else:
            assert case.rubric


def test_numeric_grading_tolerance_and_required_mentions():
    df = load_truth_frame()
    case = next(c for c in CASES if c.id == "top_region")
    truth = case.truth(df)
    assert grade_numeric(case, f"London is highest at £{truth:,.2f}", df)[0]
    assert not grade_numeric(case, f"Midlands is highest at £{truth:,.2f}", df)[0]
    assert not grade_numeric(case, "London is highest at £1.00", df)[0]
