"""Domain integrity checks; usable with unittest or pytest."""

from __future__ import annotations

import hashlib
import json
import unittest

from pydantic import ValidationError

from paper2bench.repro.models import (
    ClaimSpec,
    Dataset,
    EvidenceAnchor,
    Example,
    ExperimentSpec,
    Limits,
    Policy,
    Predictions,
    WorkloadSpec,
    canonical_bytes,
    digest,
)

HASH = "a" * 64
IMAGE = "sha256:" + "b" * 64


def valid_spec(**updates):
    data = {
        "title": "Source-backed test",
        "source_sha256": HASH,
        "dataset_sha256": "c" * 64,
        "evidence": [{"start": 0, "end": 5, "quote": "claim"}],
        "claim": {"target": 0.8},
        "workload": {"kind": "builtin_centroid"},
    }
    return ExperimentSpec.model_validate({**data, **updates})


class ModelTests(unittest.TestCase):
    def test_json_arrays_become_frozen_nested_tuples(self):
        source = {
            "train": [{"id": "tr", "x": [1, 2.0], "y": "yes"}],
            "test": [{"id": "te", "x": [2, 3], "y": "yes"}],
        }
        dataset = Dataset.model_validate_json(json.dumps(source))
        self.assertIsInstance(dataset.train, tuple)
        self.assertIsInstance(dataset.train[0].x, tuple)
        with self.assertRaises(ValidationError):
            dataset.train[0].y = "no"
        with self.assertRaises(TypeError):
            dataset.train[0].x[0] = 3

    def test_scalar_types_are_not_coerced(self):
        for value in (True, "0.5"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ClaimSpec(target=value)
        for value in (True, 1.5, "20"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ClaimSpec(target=0.5, min_test_examples=value)
        with self.assertRaises(ValidationError):
            Policy(local_only="false")
        with self.assertRaises(ValidationError):
            Example(id=1, x=(1,), y="yes")

    def test_finite_numbers_required_at_every_numeric_entry(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            for constructor in (
                lambda value=value: Example(id="a", x=(value,), y="yes"),
                lambda value=value: ClaimSpec(target=value),
                lambda value=value: ClaimSpec(target=0.5, tolerance=value),
                lambda value=value: ClaimSpec(target=0.5, confidence=value),
                lambda value=value: Limits(timeout_seconds=value),
            ):
                with self.subTest(value=value, constructor=constructor):
                    with self.assertRaises(ValidationError):
                        constructor()

    def test_unknown_fields_are_rejected(self):
        with self.assertRaises(ValidationError):
            valid_spec(unknown=True)
        with self.assertRaises(ValidationError):
            valid_spec(claim={"target": 0.8, "verified": True})

    def test_empty_and_blank_identifiers_and_labels_rejected(self):
        for value in ("", " \n\t"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                Example(id=value, x=(1,), y="yes")
            with self.subTest(value=value), self.assertRaises(ValidationError):
                Example(id="a", x=(1,), y=value)
        with self.assertRaises(ValidationError):
            Example(id="a", x=(), y="yes")

    def test_dataset_splits_and_dimensions_are_checked(self):
        row = {"id": "a", "x": [1, 2], "y": "yes"}
        valid_test = {"id": "b", "x": [2, 3], "y": "yes"}
        invalid_datasets = [
            {"train": [], "test": [valid_test]},
            {"train": [row], "test": []},
            {"train": [row], "test": [row]},
            {"train": [row, row], "test": [valid_test]},
            {"train": [row], "test": [valid_test, valid_test]},
            {"train": [row], "test": [{**valid_test, "x": [1]}]},
            {"train": [row], "test": [{**valid_test, "y": "unseen"}]},
        ]
        for data in invalid_datasets:
            with self.subTest(data=data), self.assertRaises(ValidationError):
                Dataset.model_validate(data)

    def test_anchors_use_character_offsets_and_require_matching_length(self):
        anchor = EvidenceAnchor(start=3, end=5, quote="é🧪")
        self.assertEqual(anchor.end - anchor.start, 2)
        for data in (
            {"start": -1, "end": 1, "quote": "ab"},
            {"start": 2, "end": 1, "quote": "a"},
            {"start": 0, "end": 0, "quote": ""},
            {"start": 3, "end": 9, "quote": "é🧪"},
            {"start": False, "end": 1, "quote": "a"},
        ):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                EvidenceAnchor.model_validate(data)
        with self.assertRaises(ValidationError):
            valid_spec(evidence=[])

    def test_hashes_are_full_lowercase_raw_digests(self):
        for field in ("source_sha256", "dataset_sha256", "authoring_receipt_sha256"):
            for value in ("../file", "A" * 64, "a" * 63, HASH + "\n", "sha256:" + HASH):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    valid_spec(**{field: value})
        self.assertEqual(valid_spec(authoring_receipt_sha256=HASH).authoring_receipt_sha256, HASH)

    def test_claim_probability_and_count_bounds(self):
        self.assertEqual(ClaimSpec(target=0).target, 0)
        self.assertEqual(ClaimSpec(target=1).target, 1)
        for values in (
            {"target": -0.1},
            {"target": 1.1},
            {"tolerance": 0},
            {"tolerance": -0.1},
            {"tolerance": 1.1},
            {"confidence": 0},
            {"confidence": 1},
            {"min_test_examples": 0},
            {"metric": "f1"},
        ):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                ClaimSpec.model_validate({"target": 0.5, **values})

    def test_positive_bounded_resource_limits_and_policy(self):
        for values in (
            {"timeout_seconds": 0},
            {"timeout_seconds": 3601},
            {"memory_mb": 0},
            {"memory_mb": 65537},
            {"output_bytes": 0},
            {"output_bytes": 16777217},
        ):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                Limits(**values)
        for values in (
            {"max_provider_calls": -1},
            {"max_provider_calls": 10001},
            {"max_output_tokens": 0},
            {"max_output_tokens": 32769},
        ):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                Policy(**values)

    def test_workload_fields_are_mutually_exclusive(self):
        WorkloadSpec(kind="builtin_centroid")
        WorkloadSpec(kind="python", implementation_sha256=HASH, image=IMAGE)
        WorkloadSpec(kind="python", implementation_sha256=HASH, image="repo/image@" + IMAGE)
        WorkloadSpec(kind="model_classification", provider_profile="local", model="m", prompt="p")
        invalid = [
            {"kind": "builtin_centroid", "model": "m"},
            {"kind": "builtin_centroid", "image": IMAGE},
            {"kind": "python", "implementation_sha256": HASH},
            {"kind": "python", "image": IMAGE},
            {"kind": "python", "implementation_sha256": HASH, "image": "python:latest"},
            {"kind": "python", "implementation_sha256": HASH, "image": IMAGE, "prompt": "p"},
            {"kind": "model_classification", "provider_profile": "local", "model": "m"},
            {
                "kind": "model_classification",
                "provider_profile": "local",
                "model": "m",
                "prompt": "p",
                "implementation_sha256": HASH,
            },
            {"kind": "model_classification", "provider_profile": " ", "model": "m", "prompt": "p"},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                WorkloadSpec(**values)

    def test_prediction_alignment_and_uniqueness(self):
        for values in (
            {"ids": [], "labels": []},
            {"ids": ["a"], "labels": []},
            {"ids": ["a", "a"], "labels": ["yes", "no"]},
            {"ids": ["a"], "labels": [""]},
        ):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                Predictions(**values)

    def test_canonical_json_is_sorted_compact_utf8_and_roundtrippable(self):
        value = {"z": (2, 1), "a": "é"}
        expected = '{"a":"é","z":[2,1]}'.encode()
        self.assertEqual(canonical_bytes(value), expected)
        self.assertEqual(digest(value), hashlib.sha256(expected).hexdigest())
        self.assertEqual(digest({"a": "é", "z": [2, 1]}), digest(value))
        spec = valid_spec()
        self.assertEqual(digest(spec), digest(spec.model_dump(mode="json")))
        self.assertEqual(spec, ExperimentSpec.model_validate_json(canonical_bytes(spec)))

    def test_canonical_json_rejects_lossy_and_unsupported_values(self):
        for value in ({1: "a"}, {True: "a"}, {"x": {1, 2}}, b"bytes", object()):
            with self.subTest(value=value), self.assertRaises(TypeError):
                canonical_bytes(value)
        for value in (float("nan"), {"nested": [float("inf")]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonical_bytes(value)


if __name__ == "__main__":
    unittest.main()
