import math

import pytest

from paper2bench.evaluation import EvaluationError, parse_json_object, weighted_coverage


@pytest.mark.parametrize("text", [
    "not json", 'prefix {"result": true}', '[{"result": true}]',
    '{"result": true, "result": false}', '{"nested": {"x": 1, "x": 2}}',
    '{"score": NaN}', '{"score": Infinity}', '{"score": -Infinity}', '{"score": 1e999}',
    '{"result": true} trailing', '```json\n{"x": 1}\n```\nextra',
])
def test_evaluation_parser_rejects_ambiguous_or_nonfinite_json(text):
    with pytest.raises(EvaluationError):
        parse_json_object(text)


def test_complete_json_fence_is_accepted_without_salvaging_prose():
    assert parse_json_object('```json\n{"result": false}\n```') == {"result": False}
    with pytest.raises(EvaluationError):
        parse_json_object('{"x": "' + "a" * 65536 + '"}')


def test_coverage_keeps_missing_execution_and_errors_in_denominator():
    report = weighted_coverage([
        {"id": "valid", "verdict": "pass", "weight": 1},
        {"id": "judge", "verdict": "error", "weight": 2},
        {"id": "execution", "verdict": "not_run", "weight": 1, "critical": True},
        {"id": "bad", "verdict": "fail", "weight": 1},
        {"id": "irrelevant", "verdict": "not_applicable", "weight": 999, "reason": "No training in this task"},
    ])
    assert report["applicable_weight"] == 5
    assert report["unknown_weight"] == 3
    assert report["lower_bound"] == 0.2
    assert report["upper_bound"] == 0.8
    assert report["coverage"] == 0.4
    assert report["critical_blocked"] is True
    assert report["status"] == "inconclusive"


def test_zero_applicable_weight_is_ungradable_not_a_perfect_score():
    report = weighted_coverage([])
    assert report["status"] == "ungradable"
    assert report["lower_bound"] is None
    assert report["upper_bound"] is None


def test_aggregate_finite_weights_cannot_overflow_into_nan_score():
    with pytest.raises(EvaluationError):
        weighted_coverage([
            {"id": "a", "verdict": "pass", "weight": 1e308},
            {"id": "b", "verdict": "pass", "weight": 1e308},
        ])


@pytest.mark.parametrize("criteria", [
    [{"id": "x", "verdict": "not_applicable"}],
    [{"id": "x", "verdict": "pass", "weight": math.nan}],
    [{"id": "x", "verdict": "pass", "weight": True}],
    [{"id": "x", "verdict": "pass", "weight": -1}],
    [{"id": "x", "verdict": "pass", "critical": "false"}],
    [{"id": "x", "verdict": "pass"}, {"id": "x", "verdict": "pass"}],
])
def test_invalid_coverage_records_are_errors(criteria):
    with pytest.raises(EvaluationError):
        weighted_coverage(criteria)


def test_method_failure_blocks_result_and_transitive_passes():
    report = weighted_coverage([
        {"id": "conclusion", "verdict": "pass", "depends_on": ["result"], "weight": 2},
        {"id": "result", "verdict": "pass", "depends_on": ["method"], "weight": 3},
        {"id": "method", "verdict": "fail", "critical": True},
        {"id": "independent", "verdict": "pass"},
    ])
    assert report["passed_weight"] == 1
    assert report["unknown_weight"] == 5
    assert report["applicable_weight"] == 7
    assert report["lower_bound"] == 1 / 7
    assert report["critical_blocked"] is True
    assert report["counts"]["pass"] == 1
    assert report["counts"]["not_run"] == 2
    assert report["blocked_dependencies"] == {"result": ["method"], "conclusion": ["result"]}
    assert report["status"] != "pass"


def test_all_prerequisites_must_pass_but_recorded_failures_are_retained():
    report = weighted_coverage([
        {"id": "data", "verdict": "pass"},
        {"id": "method", "verdict": "error"},
        {"id": "result", "verdict": "fail", "depends_on": ["data", "method"]},
    ])
    assert report["counts"]["fail"] == 1
    assert report["counts"]["error"] == 1
    assert report["unknown_weight"] == 1
    assert report["blocked_dependencies"] == {"result": ["method"]}


def test_successful_dependency_graph_allows_results_to_pass():
    report = weighted_coverage([
        {"id": "result", "verdict": "pass", "depends_on": ["data", "method"]},
        {"id": "method", "verdict": "pass", "depends_on": ["data"], "critical": True},
        {"id": "data", "verdict": "pass"},
    ])
    assert report["status"] == "pass"
    assert report["lower_bound"] == 1
    assert report["blocked_dependencies"] == {}


def test_not_applicable_prerequisite_does_not_provide_positive_evidence():
    report = weighted_coverage([
        {"id": "method", "verdict": "not_applicable", "reason": "No model training"},
        {"id": "result", "verdict": "pass", "depends_on": ["method"]},
    ])
    assert report["passed_weight"] == 0
    assert report["unknown_weight"] == 1
    assert report["status"] == "inconclusive"


@pytest.mark.parametrize("criteria", [
    [{"id": "x", "verdict": "pass", "depends_on": "other"}],
    [{"id": "x", "verdict": "pass", "depends_on": [1]}],
    [{"id": "x", "verdict": "pass", "depends_on": ["missing"]}],
    [{"id": "x", "verdict": "pass", "depends_on": ["x"]}],
    [{"id": "x", "verdict": "pass", "depends_on": ["y", "y"]},
     {"id": "y", "verdict": "pass"}],
    [{"id": "x", "verdict": "pass", "depends_on": ["y"]},
     {"id": "y", "verdict": "pass", "depends_on": ["x"]}],
    [{"id": "x", "verdict": "pass", "weight": 10**1000}],
    [{"id": "x", "verdict": {}}],
    [{"id": " ", "verdict": "pass"}],
])
def test_invalid_dependency_or_numeric_contracts_fail_closed(criteria):
    with pytest.raises(EvaluationError):
        weighted_coverage(criteria)


def test_long_dependency_chain_does_not_depend_on_python_recursion_depth():
    report = weighted_coverage([
        {"id": str(i), "verdict": "pass", "depends_on": [str(i - 1)] if i else []}
        for i in range(1500)
    ])
    assert report["status"] == "pass"


@pytest.mark.parametrize("values,selected", [
    ([1e308, 1e308], [0, 1]),
    ([10**1000], [0]),
])
def test_knapsack_nonfinite_aggregate_or_unrepresentable_number_is_error(values, selected):
    from paper2bench.bundles import BundleError
    from paper2bench.methods import score

    with pytest.raises(BundleError):
        score("knapsack_optimum/v1", {"weights": [1] * len(values), "values": values,
                                    "capacity": len(values)}, {"selected": selected}, {})


@pytest.mark.parametrize("values,capacity,selected", [
    ([10**16, 1], 2, [0]),
    ([1.0, 1e-13], 2, [0]),
    ([1.0, 1e-20], 2, [0]),
])
def test_knapsack_suboptimal_selection_cannot_hide_in_float_tolerance(values, capacity, selected):
    from paper2bench.methods import score

    metric = score("knapsack_optimum/v1", {"weights": [1, 1], "values": values,
                                         "capacity": capacity}, {"selected": selected}, {})
    assert metric["verdict"] == "fail"


def test_knapsack_exact_float_sums_accept_true_optimum():
    from paper2bench.methods import score

    metric = score("knapsack_optimum/v1", {"weights": [1, 1, 2], "values": [0.1, 0.2, 0.3],
                                         "capacity": 2}, {"selected": [0, 1]}, {})
    assert metric["verdict"] == "pass"
    assert metric["value"] == pytest.approx(0.3)


def test_linear_metric_overflow_is_typed_evaluation_error():
    from paper2bench.bundles import BundleError
    from paper2bench.methods import score

    with pytest.raises(BundleError, match="overflow"):
        score("linear_predictions/v1", {"test": [1]}, {"predictions": [1e308]}, {"labels": [0]})


def test_linear_float_subtraction_cannot_erase_integer_residual():
    from paper2bench.methods import score

    metric = score("linear_predictions/v1", {"test": [1]}, {"predictions": [1e16]},
                   {"labels": [10**16 + 1], "max_mse": 0})
    assert metric["value"] == 1
    assert metric["verdict"] == "fail"


def test_linear_underflow_cannot_become_a_zero_error_pass():
    from paper2bench.bundles import BundleError
    from paper2bench.methods import score

    with pytest.raises(BundleError, match="underflow"):
        score("linear_predictions/v1", {"test": [1]}, {"predictions": [1e-200]},
              {"labels": [0], "max_mse": 0})
