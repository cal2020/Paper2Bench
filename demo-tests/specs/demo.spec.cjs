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
  const copy = page.getByRole('button', { name: 'copy', exact: true }).first();
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
