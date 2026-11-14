# Demo browser regressions

These CI checks exercise the packaged demo in Chromium at desktop and narrow
mobile viewports. Every test blocks outside HTTP requests and WebSockets.
Application scripts must come from the local package; optional hosted fonts may
fall back to system fonts. No test invokes a model, starts a worker, or receives
provider credentials.

From the repository root, build the production demo first, then use Node.js 22
or newer and the checked-in test lockfile:

```bash
npm ci --prefix demo-tests
npm exec --prefix demo-tests -- playwright install --with-deps chromium
npm test --prefix demo-tests
```

The config starts an owned loopback-only server on port 4173 and stops it at the
end. An occupied port fails rather than reusing an unknown/stale server. Set
`P2B_DEMO_TEST_PORT` to choose another port; no remote base URL is accepted.

Without launching a browser, validate collection with:

```bash
npm run test:list --prefix demo-tests
```

The suite checks both browser computations, recorded protocol/broker origins,
scenario resets, copy, keyboard activation, theme persistence, paper detail
navigation and overflow. Playwright emits a JUnit report at
`demo-tests/test-results/junit.xml` and an HTML report at
`demo-tests/playwright-report/`. Failures retain a trace and screenshot under
`demo-tests/test-results/`; CI should upload both folders even on failure.
