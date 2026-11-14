"""Judge calibration accounting against independent, versioned expert labels.

Meeting an operational error target on this supplied sample is not a general
certification. The caller owns label independence, representative sampling and
review quality. Abstentions and ungraded items remain visible in coverage.
"""
from __future__ import annotations

from .statistics import finite_number
from .validation import EvaluationError


def calibration_report(records, *, judge_id, protocol_sha256, min_labeled=30,
                       max_false_accept_rate=0.05, max_false_reject_rate=0.1):
    if not isinstance(records, list) or len(records) > 100000:
        raise EvaluationError("Calibration records must be a bounded list")
    if not isinstance(judge_id, str) or not judge_id.strip():
        raise EvaluationError("Judge identity must be explicit")
    if not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64 or any(x not in "0123456789abcdef" for x in protocol_sha256):
        raise EvaluationError("Calibration protocol hash is required")
    if type(min_labeled) is not int or min_labeled < 1:
        raise EvaluationError("min_labeled must be a positive integer")
    accept_target = finite_number(max_false_accept_rate)
    reject_target = finite_number(max_false_reject_rate)
    if not 0 <= accept_target <= 1 or not 0 <= reject_target <= 1:
        raise EvaluationError("Calibration rate targets must lie in [0,1]")
    counts = {key: 0 for key in ("true_accept", "true_reject", "false_accept", "false_reject",
                                "judge_abstention", "expert_ungraded")}
    seen = set()
    for record in records:
        if not isinstance(record, dict) or set(record) - {"id", "expert", "judge", "expert_evidence"}:
            raise EvaluationError("Invalid calibration record")
        identity = record.get("id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise EvaluationError("Calibration IDs must be nonempty and unique")
        seen.add(identity)
        if (not isinstance(record.get("expert"), str) or not isinstance(record.get("judge"), str)
                or record.get("expert") not in {"pass", "fail", "inconclusive"}
                or record.get("judge") not in {"pass", "fail", "inconclusive", "error", "not_run"}):
            raise EvaluationError("Unknown calibration verdict")
        if not isinstance(record.get("expert_evidence"), str) or not record["expert_evidence"].strip():
            raise EvaluationError("Expert label provenance is required")
        expert, judge = record["expert"], record["judge"]
        if expert == "inconclusive":
            counts["expert_ungraded"] += 1
        elif judge not in {"pass", "fail"}:
            counts["judge_abstention"] += 1
        elif expert == "pass":
            counts["true_accept" if judge == "pass" else "false_reject"] += 1
        else:
            counts["true_reject" if judge == "fail" else "false_accept"] += 1
    negatives = counts["true_reject"] + counts["false_accept"]
    positives = counts["true_accept"] + counts["false_reject"]
    graded = negatives + positives
    labeled = len(records) - counts["expert_ungraded"]
    false_accept = counts["false_accept"] / negatives if negatives else None
    false_reject = counts["false_reject"] / positives if positives else None
    # Both positive and negative examples are necessary. Abstentions cannot make
    # the sample look fully calibrated by disappearing from its denominator.
    adequate = labeled >= min_labeled and negatives > 0 and positives > 0 and graded == labeled
    targets_met = adequate and false_accept <= accept_target and false_reject <= reject_target
    return {"schema_version": 1, "judge_id": judge_id, "protocol_sha256": protocol_sha256,
            "counts": counts, "total_records": len(records), "expert_labeled": labeled,
            "graded": graded, "coverage": graded / labeled if labeled else None,
            "false_accept_rate": false_accept, "false_reject_rate": false_reject,
            "agreement": (counts["true_accept"] + counts["true_reject"]) / graded if graded else None,
            "targets": {"min_labeled": min_labeled, "max_false_accept_rate": accept_target,
                        "max_false_reject_rate": reject_target},
            "status": "sample_targets_met" if targets_met else "sample_targets_failed" if adequate else "insufficient_evidence",
            "generalization": "not_certified"}
