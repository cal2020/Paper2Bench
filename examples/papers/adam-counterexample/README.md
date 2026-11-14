# Appendix A finite numerical check

This CPU experiment adapts the one-dimensional cyclic convex counterexample in
Reddi, Kale and Kumar, [On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237),
ICLR 2018 (arXiv v1). It compares Adam with AMSGrad on the same loss sequence for
30,000 rounds. It is a finite numerical check of Appendix A, not a reproduction
of Figure 1 or the paper's neural experiments, and not an asymptotic proof.

The domain is `[-1,1]`, `x_1=1`, and the gradient is `3` at rounds congruent to
`1 mod 3` and `-1` otherwise. Both algorithms use `beta1=0`, `beta2=0.1`, and
`alpha_t=0.1/sqrt(t)`, with zero initialized moments, no bias correction or
epsilon, and projection after each update. AMSGrad retains the largest past
second moment. Parameters satisfy the Appendix A construction. The paper's
Algorithm 2 also permits other schedules; this experiment compares the two
updates under this explicitly chosen shared schedule.

`solution.py` emits the complete pre-update positions `x_1,...,x_T` and the
separate post-update `x_(T+1)` for each algorithm. The controller recomputes
average regret from those positions against the best fixed comparator `x=-1`.
The primary metric is Adam's average regret minus AMSGrad's, in loss per round.
Checkpoints are fixed at 3, 30, 300, 3,000, and 30,000 rounds. A separate trusted
oracle evaluates the cyclic second moments as a geometric sum rather than
using the candidate's EMA recurrence. It checks every position and the final
update. Candidate-reported metrics cannot substitute for the raw trajectories.

Build a new frozen task from the repository root:

```sh
python -m examples.papers.adam-counterexample.build_task /tmp/adam-appendix-a-task
```

`build_task.py` hashes the candidate and inputs and freezes the independent
oracle reference before any candidate run. A replacement candidate can be
supplied with `--solution`. Its source must receive a new fidelity review bound
to its exact hash before a scientific run. The generated curator manifest has
no automatic scientific approval. A protected, reviewed `reproduction/v1`
protocol is required for scientific orchestration; `method_output/v1` checks
only the installed numerical contract.

With Docker available and `python:3.11-slim` staged locally, reproduce the
reviewed example from the repository root:

```sh
paper2bench validate --task examples/papers/adam-counterexample/task_manifest.json --executable
paper2bench validate-protocol --task examples/papers/adam-counterexample/task_manifest.json --protocol examples/papers/adam-counterexample/protocol.json
paper2bench execute --task examples/papers/adam-counterexample/task_manifest.json --evaluation-bundle examples/papers/adam-counterexample/curator_manifest.json --evaluation-profile reproduction/v1 --execution-image python:3.11-slim --runs-dir runs/adam-appendix-a
```

The reviewed curator pins the exact candidate and input hashes. Editing either
file requires rebuilding those identities and a new source-bound fidelity
review before execution; the old approval does not transfer to changed code.

The public demonstration has three distinct stages: a local foundation model
processes the real paper into an authoring draft; automated source and code
review then produce a curated executable adaptation and frozen protocol; Docker
executes the candidate while an installed controller scorer checks its complete
output. The candidate here is the curated implementation, not code autonomously
produced by the authoring model. This is a reviewed end-to-end workflow with
explicit intervention, not a fully autonomous paper reproduction.

The task needs only Python's standard library and a CPU. Its execution limit is
30 seconds with 256 MiB memory. Complete compact output stays below the
controller's 2 MiB JSON limit. Seeded re-executions check isolation and repeatable
execution, not population variability or statistical independence. All fixed
seeds, including failures, must remain visible. Do not redistribute the source PDF/full text
with this example; the source review records a hash and short factual locators.

Recorded run: three prescribed seeds passed with Adam average regret
`0.6671803186651787`, AMSGrad `0.05213238605384807`, and paired gap
`0.6150479326113306`. Maximum trace deviation was `3.3306690738754696e-16`;
the declared absolute tolerance was `1e-10`. Every container's cleanup was
confirmed. The [public receipt](../../../docs/results/adam-counterexample.json)
retains each seed and also records the earlier implementation check.

`curation.json` binds the actual local model's four authoring calls and artifact
hashes to the final task. The draft's method and plan required the corrections
listed in [SOURCE_REVIEW.md](SOURCE_REVIEW.md). The final isolated execution ran
after this authoring/curation provenance was frozen. Raw drafts remain private
and `needs_review`; the successful curated run does not score their correctness.
The reusable local profile is [runtime.local.json](runtime.local.json). After
downloading the pinned PDF privately, the authoring invocation is:

```sh
paper2bench run --pdf /path/to/1904.09237v1.pdf \
  --research-question 'Compare Adam and AMSGrad on the Appendix A cyclic-gradient counterexample over 30000 steps; finite scope only.' \
  --runtime-config examples/papers/adam-counterexample/runtime.local.json \
  --task-id adam_counterexample_real --output-dir output/adam-authoring \
  --model-runs-dir runs/adam-authoring --skip-hf-validation
```

The authoring software for this recorded case was commit `7179845fee5318bc95d3ce84118f1aa4028d470c`.
Model outputs can differ on a future run; the pinned PDF, source review and exact
curated executable are the basis for the recorded finite numerical result.
