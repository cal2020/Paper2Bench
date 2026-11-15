# Paper2Bench in Agent Lab Control Center

The importable workflow and capsule in
[`examples/control-center`](../examples/control-center/) connect this repository
to Control Center's existing repository inspection and protected verification
operations. They check the optional `paper2bench.repro` core without model calls.
Saving either asset does not execute it.

## What the current integration can do

The connected Control Center advertises a paired Mac bridge, `repo.inspect`, and
`sandbox.verify`. Its protected-verification template requires a staged image,
uses a network-disabled sandbox, and retains command output and protected-file
checks. It does not advertise a cloud-native shell. Discovery is capability
evidence, not proof that this Paper2Bench checkout has run there.

The observed app-adapter inventory has no Paper2Bench adapter. Consequently this
integration uses the existing repository workflow rather than inventing an
`app.execute` operation or changing another application's adapter. Paper2Bench's
legacy `model_broker.py` is a controller-scoped **target-model file broker**, not
a Control Center transport. Its credential isolation and runtime profiles remain
separate from the bridge used to supervise this verifier.

The observed bridge's repository list did not contain Paper2Bench. The supplied
capsule therefore has an empty `repository` field. It must be bound to the actual
enrolled repository before dispatch; an unrelated repository's ID is not a valid
substitute. No private workstation path or user-specific bridge ID is committed.

## Import the two assets

- [`paper2bench-repro-workflow.json`](../examples/control-center/paper2bench-repro-workflow.json)
  has ID `paper2bench-repro-protected-verification-v1`.
- [`paper2bench-repro-capsule.json`](../examples/control-center/paper2bench-repro-capsule.json)
  has ID `paper2bench-repro-verification-capsule-v1`.

Import them through the connected `lab_save_workflow` and `lab_save_capsule`
operations. Each JSON file is the corresponding tool's `workflow` or `capsule`
argument. Keep returned IDs, versions, and hashes as the save receipts. A save
receipt establishes storage only; it is not a run receipt.

The workflow has no model step. Foundation-model account configuration is not a
prerequisite for these deterministic checks. It inspects the selected checkout,
uses the bridge's ordinary verification approval step, and runs `sandbox.verify`.

## Bind an actual repository and dependency image

1. Make `https://github.com/cal2020/Paper2Bench` available to the owner-controlled
   Mac bridge using that bridge's repository configuration. Confirm its actual
   repository ID and intended revision through `repo.inspect`; do not guess an ID.
2. On that host, explicitly stage the verifier's dependency image from the
   intended checkout:

   ```bash
   docker build --tag paper2bench-control-center:local \
     --file examples/control-center/Dockerfile .
   docker image inspect --format '{{.Id}}' paper2bench-control-center:local
   ```

   Building downloads Python dependencies. Verification performs no installation
   or downloads. Review the base image and build dependencies under the host's
   normal supply-chain policy. The Dockerfile needs Python 3.11 or newer and
   installs the repository's optional `repro` dependencies.
3. Create a bound capsule revision with the real repository ID and the returned
   immutable image ID. The supplied image name is a build instruction, not a
   claim that the image already exists on a Mac.
4. Confirm the workflow and capsule hashes, protected file set, and fixed
   verification command before dispatch. Queue a run only when execution is
   authorized; importing these assets does not queue one.

No hosted credential, model download, SSH worker, GPU, Docker socket mount, or
foundation-model generation is required. The verifier never starts nested
containers; the outer Control Center sandbox provides its execution boundary.

## Verification and evidence

The fixed capsule command is:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python examples/control-center/verify_repro.py
```

It uses a temporary state directory and checks the actual optional CLI schema
entrypoint; source-linked frozen identities; deterministic classification and
knapsack execution; trusted scoring; idempotent run identity; portable score
recomputation; changed-setting identity; key conflicts; cancellation fencing; and
artifact corruption rejection. Model calls and generated-code execution are
explicitly forbidden in this verifier. Its JSON report lists each completed
check. It does not replace the full unit, Docker, provider, or remote-worker suites.

A passing integration requires the matching native receipt with a zero command
exit code, an untruncated successful check report, `protectedUnchanged: true`, and
the intended repository revision, resolved image identity, and network setting.
Retain the workflow/capsule hashes and run ID alongside the receipt. Inspect
failure, timeout, cancellation, blocked, or missing-output states as those states;
an empty or unfinished result does not pass.

The checks establish synthetic software behavior. They do not establish
published-paper reproduction, implementation fidelity to a paper, foundation-model
quality, hosted provider readiness, a deployed website, or independent runtime
attestation. Portable archive replay reports authentication as `not_verified`
even when artifact hashes and recomputed scores pass.

## Other Paper2Bench workflows

The pre-existing `paper2bench` CLI and runtime profiles remain available alongside
`python -m paper2bench.repro` / `paper2bench-repro`. For actual local or hosted
authoring, configure explicit model/provider bindings and budgets using the
appropriate runtime. Do not transfer a successful model-free Control Center
check into an inference-success claim. GPU, SSH, and hosted execution require
their own worker/provider evidence.

## Portfolio Project Control Center import

The separate Portfolio Project Control Center accepts a workspace JSON object
with a `projects` array. The prepared
[`portfolio-project-update.json`](../examples/control-center/portfolio-project-update.json)
contains exactly one full-schema project, with stable ID `paper2bench`. It is a
reviewable import artifact, not evidence that the live portfolio was changed.

The live authenticated Portfolio API request was blocked by HTTP 403 with
Cloudflare `browser_signature_banned`. The denied route was not retried or
bypassed. Use the application's supported import UI when available:

1. Select the prepared JSON file and inspect the one-project preview.
2. Choose **Skip existing projects; add new IDs only** (`add_only`) for the
   initial import. This skips an existing `paper2bench` record instead of
   overwriting its current fields, evidence, tasks, or notes.
3. If that ID already exists, review the live record before any later merge.
   The application's `merge` mode replaces project fields while retaining prior
   history and evidence; it is not an automatic field-by-field conflict resolution.
4. Confirm the imported result. Importing recorded evidence performs no new
   verification. Keep the live mutation blocked until an actual import receipt
   or inspected UI result establishes it.

The record separates the final local gate of 1308 Python tests plus 93 subtests
and 38 browser checks from the historical 577-test framework receipt, original
642-test Repro receipt, and later reported 646-test standalone gate. New GitHub
CI, merge, and demo deployment remain pending until their exact final receipts
are inspected. Public links to new source paths identify intended publication
locations; they do not establish that the pending change is already on `main`.
No unrelated project state, private Mac identifier, or credential is included.
