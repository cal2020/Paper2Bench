"""Scientific verdict and trusted scoring checks."""

from __future__ import annotations

import math
import unittest

from paper2bench.repro.evaluation import evaluate
from paper2bench.repro.models import ClaimSpec, Dataset, Example, Predictions, canonical_bytes


def case(count=100, correct=80):
    dataset = Dataset(
        train=(Example(id="train_yes", x=(0,), y="yes"), Example(id="train_no", x=(1,), y="no")),
        test=tuple(Example(id=f"test_{i}", x=(float(i),), y="yes") for i in range(count)),
    )
    predictions = Predictions(
        ids=tuple(example.id for example in dataset.test),
        labels=tuple("yes" if i < correct else "no" for i in range(count)),
    )
    return dataset, predictions


class EvaluationTests(unittest.TestCase):
    def test_known_wilson_interval_and_supported_equivalence(self):
        dataset, predictions = case()
        result = evaluate(dataset, predictions, ClaimSpec(target=0.8, tolerance=0.1))
        self.assertEqual(result["accuracy"], 0.8)
        self.assertEqual(result["correct"], 80)
        self.assertEqual(result["sample_count"], 100)
        self.assertAlmostEqual(result["confidence_interval"]["lower"], 0.7111708344, places=9)
        self.assertAlmostEqual(result["confidence_interval"]["upper"], 0.8666330667, places=9)
        self.assertEqual(result["verdict"], "supported")

    def test_disjoint_interval_is_not_supported(self):
        dataset, predictions = case(correct=50)
        result = evaluate(dataset, predictions, ClaimSpec(target=0.9, tolerance=0.05))
        self.assertEqual(result["verdict"], "not_supported")

    def test_overlap_without_containment_is_inconclusive(self):
        dataset, predictions = case()
        result = evaluate(dataset, predictions, ClaimSpec(target=0.9, tolerance=0.1))
        self.assertEqual(result["verdict"], "inconclusive")

    def test_minimum_samples_overrides_apparent_support_or_refutation(self):
        for correct, claim in (
            (1, ClaimSpec(target=1, tolerance=1, min_test_examples=20)),
            (0, ClaimSpec(target=1, tolerance=0.01, min_test_examples=20)),
        ):
            dataset, predictions = case(count=1, correct=correct)
            result = evaluate(dataset, predictions, claim)
            self.assertEqual(result["verdict"], "inconclusive")
            self.assertFalse(result["minimum_sample_count_met"])

    def test_zero_and_perfect_accuracy_have_exact_probability_endpoints(self):
        for correct, endpoint in ((0, "lower"), (100, "upper")):
            dataset, predictions = case(correct=correct)
            result = evaluate(dataset, predictions, ClaimSpec(target=correct / 100, tolerance=0.1))
            self.assertEqual(result["confidence_interval"][endpoint], correct / 100)
            self.assertEqual(result["verdict"], "supported")

    def test_criterion_band_is_clipped(self):
        dataset, predictions = case()
        low = evaluate(dataset, predictions, ClaimSpec(target=0.01, tolerance=0.1))
        high = evaluate(dataset, predictions, ClaimSpec(target=0.99, tolerance=0.1))
        self.assertEqual(low["criterion_band"]["lower"], 0)
        self.assertEqual(high["criterion_band"]["upper"], 1)

    def test_ordered_audit_matches_ids_not_prediction_order(self):
        dataset, predictions = case(count=4, correct=2)
        reversed_predictions = Predictions(
            ids=predictions.ids[::-1], labels=predictions.labels[::-1]
        )
        claim = ClaimSpec(target=0.5)
        normal = evaluate(dataset, predictions, claim)
        reversed_result = evaluate(dataset, reversed_predictions, claim)
        self.assertEqual(normal, reversed_result)
        self.assertEqual([row["id"] for row in normal["per_item"]], list(predictions.ids))
        self.assertEqual([row["correct"] for row in normal["per_item"]], [True, True, False, False])

    def test_missing_extra_and_training_ids_rejected(self):
        dataset, predictions = case(count=2, correct=2)
        invalid = (
            Predictions(ids=predictions.ids[:1], labels=predictions.labels[:1]),
            Predictions(ids=predictions.ids + ("extra",), labels=predictions.labels + ("yes",)),
            Predictions(ids=("train_yes", "test_1"), labels=("yes", "yes")),
        )
        for prediction in invalid:
            with self.subTest(prediction=prediction), self.assertRaises(ValueError):
                evaluate(dataset, prediction, ClaimSpec(target=1))

    def test_unknown_output_label_rejected(self):
        dataset, predictions = case(count=1, correct=1)
        invalid = Predictions(ids=predictions.ids, labels=("invented",))
        with self.assertRaises(ValueError):
            evaluate(dataset, invalid, ClaimSpec(target=1))

    def test_higher_confidence_interval_contains_lower_confidence_interval(self):
        dataset, predictions = case()
        low = evaluate(dataset, predictions, ClaimSpec(target=0.8, confidence=0.8))
        high = evaluate(dataset, predictions, ClaimSpec(target=0.8, confidence=0.99))
        self.assertLess(high["confidence_interval"]["lower"], low["confidence_interval"]["lower"])
        self.assertGreater(
            high["confidence_interval"]["upper"], low["confidence_interval"]["upper"]
        )

    def test_extreme_valid_confidence_remains_finite(self):
        dataset, predictions = case(count=1, correct=1)
        result = evaluate(
            dataset, predictions, ClaimSpec(target=1, confidence=math.nextafter(1, 0))
        )
        for key in ("lower", "upper"):
            self.assertTrue(math.isfinite(result["confidence_interval"][key]))
            self.assertGreaterEqual(result["confidence_interval"][key], 0)
            self.assertLessEqual(result["confidence_interval"][key], 1)
        canonical_bytes(result)

    def test_interval_touching_band_is_inconclusive_not_disjoint(self):
        dataset, predictions = case(count=1, correct=1)
        first = evaluate(dataset, predictions, ClaimSpec(target=1, min_test_examples=1))
        boundary = first["confidence_interval"]["lower"]
        result = evaluate(
            dataset,
            predictions,
            ClaimSpec(target=boundary / 2, tolerance=boundary / 2, min_test_examples=1),
        )
        self.assertEqual(result["verdict"], "inconclusive")

    def test_numerical_agreement_does_not_assert_reproduction(self):
        dataset, predictions = case()
        result = evaluate(dataset, predictions, ClaimSpec(target=0.8))
        self.assertEqual(result["criterion"], "accuracy_equivalence")
        self.assertEqual(result["implementation_fidelity"], "unverified")
        self.assertEqual(result["protocol_fidelity"], "unverified")
        self.assertIn("does not establish", result["interpretation"])
        self.assertIn("independent", result["interpretation"])


if __name__ == "__main__":
    unittest.main()
