const fs = require('node:fs');
const path = require('node:path');
const { test: base, expect } = require('@playwright/test');

const sourceEvidence = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../../docs/verification.json'), 'utf8'));

// Every test runs with outside HTTP requests blocked. An offsite script request
// is a failure even if some fallback makes the page appear to work. Optional
// remote fonts may fail; browser and system font fallbacks remain usable.
const test = base.extend({
  networkGuard: [async ({ context, page, baseURL }, use) => {
    const origin = new URL(baseURL).origin;
    const apiRequests = [];
    const outsideScripts = [];
    const errors = [];
    const missingAssets = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('response', response => {
      if (new URL(response.url()).origin === origin && response.status() >= 400 &&
          ['document', 'script', 'stylesheet'].includes(response.request().resourceType())) {
        missingAssets.push(`${response.status()} ${response.url()}`);
      }
    });
    await context.route('**/*', async route => {
      const request = route.request();
      const url = new URL(request.url());
      if (['fetch', 'xhr', 'websocket'].includes(request.resourceType()) || !['GET', 'HEAD'].includes(request.method())) {
        apiRequests.push(`${request.method()} ${request.url()}`);
      }
      if (url.origin !== origin) {
        if (request.resourceType() === 'script') outsideScripts.push(request.url());
        await route.abort('blockedbyclient');
      } else {
        await route.continue();
      }
    });
    await context.routeWebSocket(/.*/, socket => {
      apiRequests.push(`WebSocket ${socket.url()}`);
      socket.close();
    });
    await use({ apiRequests, outsideScripts, errors, missingAssets });
    expect(errors, 'No uncaught application errors').toEqual([]);
    expect(missingAssets, 'All packaged application assets resolve').toEqual([]);
    expect(outsideScripts, 'Application JavaScript is self-contained').toEqual([]);
    expect(apiRequests, 'Static demo never dispatches model or compute API requests').toEqual([]);
  }, { auto: true }],
});

async function openDemo(page) {
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Explore the execution evidence.' })).toBeVisible();
}

function scenario(page, title) {
  return page.getByRole('group', { name: 'Choose an execution scenario' }).getByRole('button', { name: new RegExp('^' + title) });
}

test('packaged app loads without outside JavaScript and exposes the correct version', async ({ page }) => {
  await openDemo(page);
  await expect(page).toHaveTitle(/Paper2Bench/);
  const home = page.getByRole('link', { name: 'Paper2Bench home' });
  await expect(home).toContainText(`v${sourceEvidence.version}`);
  await expect(page.getByRole('link', { name: 'GitHub repository' })).toHaveAttribute('href', 'https://github.com/cal2020/Paper2Bench');
  await expect(page.getByRole('link', { name: 'View the sanitized evidence summary ↗' })).toHaveAttribute('href', 'verification.json');
});

test('linear fitting computes the held-out predictions in the browser', async ({ page }) => {
  await openDemo(page);
  await page.getByRole('button', { name: 'Run browser check', exact: true }).click();
  const result = page.getByRole('status');
  await expect(result).toContainText('Computed in this browser');
  await expect(result).toContainText('MSE 0');
  await expect(result).toContainText('Predictions: 9, 11, 21.');
});

test('symbolic optimization computes a feasible optimum and clears the old result', async ({ page }) => {
  await openDemo(page);
  await page.getByRole('button', { name: 'Run browser check', exact: true }).click();
  await scenario(page, 'Symbolic optimization').click();
  const result = page.getByRole('status');
  await expect(result).toContainText('Choose a scenario');
  await expect(result).not.toContainText('MSE 0');
  await page.getByRole('button', { name: 'Run browser check', exact: true }).click();
  await expect(result).toContainText('Computed in this browser');
  await expect(result).toContainText('Objective 12');
  await expect(result).toContainText('Weight 8 / 8. All 16 subsets checked.');
});

test('scientific panel labels its saved synthetic evidence and uncertainty', async ({ page }) => {
  await openDemo(page);
  await scenario(page, 'Reviewed protocol').click();
  await page.getByRole('button', { name: 'Inspect recorded run', exact: true }).click();
  const result = page.getByRole('status');
  await expect(result).toContainText('Recorded local Docker run');
  await expect(result).toContainText('3 seeds · 3 passing criteria');
  await expect(result).toContainText('Statistical uncertainty: not assessed.');
  await expect(page.getByText('Synthetic contract and author review only.', { exact: false })).toBeVisible();
});

test('model broker panel inspects a receipt without making a model request', async ({ page }) => {
  await openDemo(page);
  await scenario(page, 'Local model broker').click();
  await page.getByRole('button', { name: 'Inspect recorded run', exact: true }).click();
  const result = page.getByRole('status');
  await expect(result).toContainText('Recorded local Llama + Docker fixture');
  await expect(result).toContainText('1 model call · 0 on replay');
  await expect(page.getByText('This site makes no model calls;', { exact: false })).toBeVisible();
});

test('copy command is available through the real clipboard control', async ({ page }) => {
  await openDemo(page);
  // The accessible name changes to "✓ copied" after success. Keep selecting
  // the same first hero control rather than switching to the next "copy".
  const copy = page.getByRole('banner').getByRole('button').first();
  await expect(copy).toHaveText('copy');
  await copy.click();
  await expect(copy).toHaveText('✓ copied');
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe('pip install git+https://github.com/cal2020/Paper2Bench.git');
});

test('scenario controls support keyboard activation and expose pressed state', async ({ page }) => {
  await openDemo(page);
  const linear = scenario(page, 'Linear fitting');
  const symbolic = scenario(page, 'Symbolic optimization');
  await linear.press('Enter');
  await expect(linear).toHaveAttribute('aria-pressed', 'true');
  await linear.press('Tab');
  await expect(symbolic).toBeFocused();
  await symbolic.press('Space');
  await expect(symbolic).toHaveAttribute('aria-pressed', 'true');
  await expect(linear).toHaveAttribute('aria-pressed', 'false');
  await expect(page.getByRole('heading', { name: 'Symbolic optimization', exact: true })).toBeVisible();
});

test('theme toggle survives a reload', async ({ page }) => {
  await openDemo(page);
  await page.getByRole('button', { name: 'Switch to dark mode' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-mode', 'dark');
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-mode', 'dark');
  await expect(page.getByRole('button', { name: 'Switch to light mode' })).toBeVisible();
});

test('CGCNN detail preserves the draft warning and paper walkthrough', async ({ page }) => {
  await page.goto('/example.html?paper=cgcnn');
  await expect(page.getByRole('heading', { name: 'CGCNN', exact: true })).toBeVisible();
  await expect(page.getByText('Illustrative paper draft.', { exact: true })).toBeVisible();
  await expect(page.getByText('This upstream-derived content has not been validated as a reproduction', { exact: false })).toBeVisible();
  await page.getByRole('button', { name: '02 Decompose the science' }).click();
  await expect(page.getByRole('button', { name: '02 Decompose the science' })).toHaveClass(/active/);
  await expect(page.getByText('CGCNN · illustrative draft', { exact: true })).toBeVisible();
});

test('main and detail pages fit the viewport without document overflow', async ({ page }) => {
  for (const url of ['/', '/example.html?paper=cgcnn']) {
    await page.goto(url);
    await expect(page.getByRole('link', { name: 'Paper2Bench home' })).toBeVisible();
    const dimensions = await page.evaluate(() => ({
      viewport: document.documentElement.clientWidth,
      document: document.documentElement.scrollWidth,
    }));
    expect(dimensions.document, `${url} must not scroll horizontally`).toBeLessThanOrEqual(dimensions.viewport + 1);
  }
});

test('repro runtime workbench selects boundaries without dispatching work', async ({ page }) => {
  await openDemo(page);
  const workbench = page.locator('#repro-runtime');
  await expect(workbench.getByRole('heading', { name: 'A reproducibility workbench.' })).toBeVisible();
  const controls = workbench.getByRole('group', { name: 'Choose a reproducibility boundary' });
  const frozen = controls.getByRole('button', { name: /Frozen experiments$/ });
  const workers = controls.getByRole('button', { name: /Fenced remote workers$/ });
  await expect(frozen).toHaveAttribute('aria-pressed', 'true');
  await workers.click();
  await expect(workers).toHaveAttribute('aria-pressed', 'true');
  await expect(frozen).toHaveAttribute('aria-pressed', 'false');
  const panel = workbench.locator('#repro-panel');
  await expect(panel.getByRole('heading', { name: 'Fenced remote workers', exact: true })).toBeVisible();
  await expect(panel).toContainText('Worker scope does not grant administrator routes');
  await expect(panel).toContainText('not hardware attestation');
  await expect(panel.getByRole('link', { name: 'Remote protocol ↗' })).toHaveAttribute('href', 'https://github.com/cal2020/Paper2Bench/blob/main/paper2bench/repro/remote.py');
  await expect(workbench.getByRole('textbox')).toHaveCount(0);
  await expect(workbench).toContainText('This static walkthrough never connects to a runtime');
  await expect(workbench).toContainText('Use an active Python 3.11+ virtual environment');
});

test('repro boundary controls support keyboard navigation and disclose scoring scope', async ({ page }) => {
  await openDemo(page);
  const workbench = page.locator('#repro-runtime');
  const controls = workbench.getByRole('group', { name: 'Choose a reproducibility boundary' });
  const frozen = controls.getByRole('button', { name: /Frozen experiments$/ });
  const scoring = controls.getByRole('button', { name: /Independent scoring$/ });
  await frozen.focus();
  await frozen.press('Tab');
  await expect(scoring).toBeFocused();
  await scoring.press('Space');
  await expect(scoring).toHaveAttribute('aria-pressed', 'true');
  const panel = workbench.locator('#repro-panel');
  await expect(panel.getByRole('heading', { name: 'Independent scoring', exact: true })).toBeVisible();
  await expect(panel).toContainText('Candidates return predictions or selected item indices');
  await expect(panel).toContainText('a synthetic software fixture');
  await expect(panel.getByRole('link', { name: 'Inspect checks ↗' })).toHaveAttribute('href', 'https://github.com/cal2020/Paper2Bench/blob/main/tests/repro/test_optimization.py');
});

test('repro uncertainty, quotas and container limits stay explicit', async ({ page }) => {
  await openDemo(page);
  const workbench = page.locator('#repro-runtime');
  const controls = workbench.getByRole('group', { name: 'Choose a reproducibility boundary' });
  const panel = workbench.locator('#repro-panel');
  await controls.getByRole('button', { name: /Multi-seed replay$/ }).click();
  await expect(panel).toContainText('Inconclusive; no successful-only selection');
  await expect(panel).toContainText('not publisher authentication or verified seed consumption');
  await controls.getByRole('button', { name: /Shared model quotas$/ }).click();
  await expect(panel).toContainText('Unknown without a verified price schedule');
  await expect(panel).toContainText('not a promise about an external provider’s final invoice');
  await controls.getByRole('button', { name: /Crash and parser boundaries$/ }).click();
  await expect(panel).toContainText('The in-container deadline remains active');
  await expect(panel).toContainText('OCR and equation reconstruction are unsupported');
  await expect(panel.getByRole('link', { name: 'Isolated PDF parser ↗' })).toHaveAttribute('href', 'https://github.com/cal2020/Paper2Bench/blob/main/paper2bench/repro/parsers.py');
});

test('selected repro assets resolve and the evidence ZIP is labeled honestly', async ({ page, baseURL }) => {
  await openDemo(page);
  const workbench = page.locator('#repro-runtime');
  for (const [label, path] of [
    ['Runtime verification receipt ↗', 'repro/verification-v0.2.json'],
    ['Synthetic optimization report ↗', 'repro/optimization-v0.2.json'],
    ['Synthetic multi-seed study ↗', 'repro/seeded-study-v0.2.json'],
    ['Inspect selected asset hashes ↗', 'repro/manifest.json'],
  ]) {
    await expect(workbench.getByRole('link', { name: label, exact: true })).toHaveAttribute('href', path);
    const response = await page.request.get(new URL(path, baseURL).href);
    expect(response.ok(), path + ' is packaged').toBeTruthy();
    expect(await response.json()).toBeTruthy();
  }
  await expect(workbench).toContainText('Sanitized reports and hashes; not a full replay bundle.');
  const downloadEvent = page.waitForEvent('download');
  await workbench.getByRole('link', { name: 'Download selected evidence ZIP ↓', exact: true }).click();
  const download = await downloadEvent;
  expect(download.suggestedFilename()).toBe('paper2bench-repro-selected-evidence-v0.2.zip');
  const zip = fs.readFileSync(await download.path());
  expect(zip.subarray(0, 4).equals(Buffer.from([0x50, 0x4b, 0x03, 0x04]))).toBeTruthy();
  await expect(workbench.getByRole('link', { name: 'Read the reproducibility runtime guide ↗' })).toHaveAttribute('href', 'https://github.com/cal2020/Paper2Bench/blob/main/docs/REPRODUCIBILITY.md');
});

test('repro workbench fits each viewport with every boundary selected', async ({ page }) => {
  await openDemo(page);
  const workbench = page.locator('#repro-runtime');
  const controls = workbench.getByRole('group', { name: 'Choose a reproducibility boundary' });
  for (const name of ['Frozen experiments', 'Independent scoring', 'Fenced remote workers', 'Crash and parser boundaries', 'Shared model quotas', 'Multi-seed replay']) {
    await controls.getByRole('button', { name: new RegExp(name + '$') }).click();
    await expect(workbench.locator('#repro-panel').getByRole('heading', { name, exact: true })).toBeVisible();
    const dimensions = await page.evaluate(() => ({
      viewport: document.documentElement.clientWidth,
      document: document.documentElement.scrollWidth,
    }));
    expect(dimensions.document, name + ' must fit the viewport').toBeLessThanOrEqual(dimensions.viewport + 1);
  }
});
