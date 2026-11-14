# Scientific protocol integration

The protocol engine provides `reproduction/v1` and `transfer/v1` decisions for
an explicit, frozen question, dataset, code submission and evaluation procedure.
A curator must supply the protocol, citations and method review. The engine does
not infer scientific fidelity from a matching score or a fluent conclusion.

The wire format is documented by
`paper2bench/schemas/scientific-protocol-v1.json`. Runtime validation additionally
checks identities, dependencies and evidence provenance, which JSON Schema alone
cannot enforce. `tests/test_science_protocol.py` contains complete synthetic
fixtures covering both successful and contradictory results; their dummy digests
are test identities, not evidence that a real paper was reproduced.

Controller integration has two phases:

```python
from paper2bench.evaluation.science import freeze_protocol, evaluate_protocol

code_sha = task_manifest["files"][task_manifest["entrypoint"]]
frozen = freeze_protocol(protocol, task_manifest=task_manifest,
                         submission_sha256=code_sha)
# Persist frozen.as_dict() and frozen.sha256 before candidate execution.
# Execute the sealed submission in a clean worker. Recompute metrics using
# installed, protected scoring code and evaluator-controlled inputs/labels.
report = evaluate_protocol(frozen, task_manifest=task_manifest,
                           submission_sha256=code_sha,
                           measurement_evidence=controller_evidence)
```

The evidence envelope is built by trusted controller code. Candidate files must
never be loaded directly as this envelope, even if they contain matching origin
strings. It binds the exact protocol, task, submission and data hashes; records
the pre-execution lock; links clean execution evidence; and includes one checked
metric record per prescribed seed. Every successful observation carries units,
sample count, output-artifact digest and evidence references. Failed, missing or
duplicate seed observations cannot be silently dropped or selectively pooled.

Critical result criteria depend on clean execution and a curator method review
of the exact submission. The report keeps implementation fidelity, execution,
empirical agreement and transfer distinct. A faithful contradictory result can
pass fidelity/execution and fail agreement. A transfer task uses its changed
protocol, so a reversed source-paper effect is not automatically incorrect.

Absolute/relative tolerances are point-agreement rules. Interval equivalence
requires a prespecified bounded range and an independence assumption; its
Hoeffding interval must fit wholly inside the declared equivalence margin.
Overlapping intervals are inconclusive, not equivalent. Bonferroni correction
accounts for multiple metric intervals. These bounds quantify the stated run
variability, not generalization across papers, people or datasets.

`calibration_report` in `paper2bench.evaluation.calibration` compares versioned
judge decisions against provenance-linked expert labels. It preserves abstention
coverage and distinguishes insufficient evidence from meeting sample-level error
targets. It never labels a small supplied sample as general judge certification.
