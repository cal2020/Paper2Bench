"""Protected evaluation primitives. Candidate data is never executable input."""

from .validation import EvaluationError, parse_json_object, weighted_coverage

__all__ = ["EvaluationError", "parse_json_object", "weighted_coverage"]
