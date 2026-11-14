# Demo deployment

The demo is a static, interactive presentation of Paper2Bench 0.4.0a2. Recorded
synthetic results are labeled; this deployment does not run candidate code,
accept provider credentials, or expose the controller runtime.

Source: `docs/` in [cal2020/Paper2Bench](https://github.com/cal2020/Paper2Bench).
The Site's stable identity and static directory are tracked in
[.openai/hosting.json](.openai/hosting.json). Access remains owner-private.

The production build bundles React and the existing JSX into local JavaScript.
It does not fetch application libraries or compile JSX in the visitor's browser.
Hosted fonts are optional; system-font fallbacks work when outside requests are
blocked. The original `docs/` HTML remains available for source inspection.

Use Python 3.10+ and Node.js 24 (CI pins 24.22.2). Install the two checked-in npm lockfiles from the
repository root:

```bash
npm ci
npm ci --prefix demo-tests
python3 scripts/build_demo.py
python3 -m http.server 8765 --bind 127.0.0.1 --directory deployment/demo/dist
```

If an older `node` is first on PATH, select the installed current executable with
`PAPER2BENCH_NODE`. For npm, put that executable's directory first on PATH.

Development builds disclose their dirty state in `dist/build-provenance.json`.
Release builds require a clean checkout and bind the output to the full GitHub
commit/tree, source-file hashes, dependency lockfile, and Site identity:

```bash
python3 scripts/build_demo.py --release
python3 scripts/package_demo.py --source deployment/demo \
  --output /tmp/paper2bench-demo.tar.gz --expected-commit "$(git rev-parse HEAD)"
```

The packer verifies every asset and its exact file-set closure, checks source
identities against Git objects, and emits a deterministic archive containing
`.openai/hosting.json` and `dist/`. A checksum establishes integrity and identity;
it is not an independent attestation of scientific validity.

## GitHub verification and publication

[Demo artifact](../../.github/workflows/demo.yml) builds with `npm ci`, validates
the release, and runs 20 Chromium checks across desktop and mobile while outside
requests are blocked. It tests computations, evidence labels, navigation, copy,
keyboard access, theme persistence, and page overflow. Only a successful job saves
`paper2bench-demo-FULL_COMMIT_SHA`; reports and failure traces are separate
artifacts. Artifacts expire after 14 days; browser reports after 7 days.

GitHub builds and tests with read-only repository permissions. It receives no
Site or model credentials and never publishes a pull request automatically.
Publication uses the exact artifact from a successful push run for the intended
`main` commit:

1. Confirm the workflow conclusion, event, branch, and full commit SHA. Download
   that run's matching demo artifact and verify its SHA-256 file.
2. Extract and validate into an empty directory using
   `python3 scripts/package_demo.py --extract ARCHIVE --extract-to DIRECTORY --expected-commit FULL_SHA`.
3. Open this existing Site through the Sites source workflow using a fresh
   short-lived credential. Copy the validated `.openai/hosting.json` and `dist/`
   into its returned source checkout, preserving the Site's identity and access.
4. Let the source workflow commit/push and package those exact bytes. Save and
   deploy its returned Site commit, then confirm the terminal deployment result.

The Site source commit and GitHub application commit are different identities;
`dist/build-provenance.json` records the bridge. Never rebuild or modify an
artifact between browser verification and publication. Never copy `.venv`,
`runs`, `.env`, model credentials, Git metadata, or private runtime state into
the assets. Keep credentials in memory and pass them only through the Sites
workflow's hidden stdin; do not store them in GitHub secrets or repository files.

Production: https://paper2bench-portable-demo.escape4242.chatgpt.site

The Python CLI/SDK is installed from GitHub and executes on an operator's local
or SSH Docker worker. This static deployment has no compute API or job service.
