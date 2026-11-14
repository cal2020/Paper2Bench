import pytest

from paper2bench.evaluation.calibration import calibration_report
from paper2bench.evaluation.validation import EvaluationError


def item(identity, expert="pass", judge="pass"):
    return {"id": str(identity), "expert": expert, "judge": judge,
            "expert_evidence": f"review/{identity}.json"}


def report(items, **kwargs):
    return calibration_report(items, judge_id="judge-version-1", protocol_sha256="a" * 64, **kwargs)


def test_confusion_counts_and_error_rates_are_explicit():
    result = report([item(1), item(2, "fail", "fail"), item(3, "fail", "pass"), item(4, "pass", "fail")], min_labeled=4)
    assert result["false_accept_rate"] == 0.5
    assert result["false_reject_rate"] == 0.5
    assert result["agreement"] == 0.5
    assert result["status"] == "sample_targets_failed"
    assert result["generalization"] == "not_certified"


def test_small_or_one_sided_sets_cannot_meet_calibration_targets():
    assert report([item(1), item(2, "fail", "fail")])["status"] == "insufficient_evidence"
    assert report([item(i) for i in range(30)])["status"] == "insufficient_evidence"


def test_abstentions_and_ungraded_references_do_not_disappear():
    result = report([item(1), item(2, "fail", "fail"), item(3, "fail", "error"), item(4, "inconclusive", "pass")], min_labeled=2)
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["counts"]["expert_ungraded"] == 1
    assert result["counts"]["judge_abstention"] == 1
    assert result["status"] == "insufficient_evidence"


def test_sample_targets_met_is_not_general_certification():
    result = report([item(i, "pass" if i % 2 else "fail", "pass" if i % 2 else "fail") for i in range(30)])
    assert result["status"] == "sample_targets_met"
    assert result["generalization"] == "not_certified"


@pytest.mark.parametrize("records", [[item(1), item(1)], [dict(item(1), expert_evidence="")], [dict(item(1), judge=True)]])
def test_invalid_calibration_records_are_rejected(records):
    with pytest.raises(EvaluationError):
        report(records)
