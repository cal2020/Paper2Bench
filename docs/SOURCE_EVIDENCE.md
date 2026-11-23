# Actual source-check evidence (manual main verification)

The existing Demo artifact workflow already runs build admission and archive-integrity tests. This pilot measures those tests plus six cheap receipt-safety checks instead of adding another job, matrix, model call, Docker run or installation. Run the workflow explicitly from the Actions tab using **Run workflow**. It does not start on pushes, pull requests, or tags. The compact evidence artifact is retained only for a manual run on `refs/heads/main`. A branch run can test the branch but does not create this main-only receipt. Earlier push-run receipts describe only their own historical commits.

The evidence command requires a clean committed checkout and records the exact full commit, canonical portfolio project ID `paper2bench`, sourceSnapshotId `cal2020/Paper2Bench@SHA`, execution origin, actual exit code, measured check-process wall duration and JUnit aggregate counts. A GitHub run link identifies execution; green workflow status alone is not used as evidence. The test scope is only build admission and archive integrity, with deliberate test fixtures. It does not prove model quality, scientific results, browser behavior, deployment, or all repository tests. Hours and tokens remain null. Missing JUnit evidence cannot pass.

Only source-evidence.json is uploaded. Raw JUnit, test logs, environment variables, account records and local absolute paths are not exported. An explicit immutable source path allowlist is hashed: this script, the build and packaging scripts, their two test files plus receipt safety tests, this guide, and the demo workflow. Missing files or symlinks fail; no recursive archive is created. This receipt hashes reviewed public source, not a deployable demo artifact. Existing demo artifact and browser report handling are unchanged and separate.

GitHub permission remains contents: read; checkout does not persist credentials. No extra secret, broad token, write permission, automatic trigger, merge or deployment is enabled by the checker. Existing pinned action revisions are reused. Retention is 14 days. An unavailable runner yields no receipt, not a successful zero-duration run.

Local verification, from a clean source checkout with pytest installed:

```
python scripts/source_evidence.py --output /path/to/new-evidence-directory
```

The destination must not exist. The command uses a bounded 180-second check timeout. A failed/timed-out check produces a failure receipt when possible and exits nonzero. Old reports are never overwritten. For public portfolio ingestion, validate the exact source commit and artifact hashes, keep test scope and timestamp, and reconcile this project identity with the existing Control Center. This hook does not update that site or Hours Atlas automatically. A separately reviewed consumer can use the actual execution evidence without inventing usage totals.
