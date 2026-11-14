const fs = require('node:fs');
const path = require('node:path');
const { test: base, expect } = require('@playwright/test');

const receiptPath = path.resolve(__dirname, '../../docs/results/adam-counterexample.json');
const receipt = fs.existsSync(receiptPath) ? JSON.parse(fs.readFileSync(receiptPath, 'utf8')) : null;

// A published paper result must work from the packaged static evidence alone.
// Treat model calls, compute calls, offsite scripts, and missing assets as errors.
const test = base.extend({
  staticEvidenceGuard: [async ({ context, page, baseURL }, use) => {
    const origin = new URL(baseURL).origin;
    const errors = [];
    const liveRequests = [];
    const outsideScripts = [];
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
        liveRequests.push(`${request.method()} ${request.url()}`);
      }
      if (url.origin !== origin) {
        if (request.resourceType() === 'script') outsideScripts.push(request.url());
        await route.abort('blockedbyclient');
      } else {
        await route.continue();
      }
    });
    await context.routeWebSocket(/.*/, socket => {
      liveRequests.push(`WebSocket ${socket.url()}`);
      socket.close();
    });
    await use();
    expect(errors, 'No uncaught paper-study application errors').toEqual([]);
    expect(missingAssets, 'Every packaged paper-study asset resolves').toEqual([]);
    expect(outsideScripts, 'Paper-study JavaScript is self-contained').toEqual([]);
    expect(liveRequests, 'Inspecting recorded evidence never calls a model or compute API').toEqual([]);
  }, { auto: true }],
});

async function openPaper(page) {
  expect(receipt, 'A real, sanitized receipt is required before publishing this case study').not.toBeNull();
  expect(receipt.schema_version).toBe(1);
  await page.goto('/#real-paper');
  const section = page.locator('#real-paper');
  await expect(section.getByRole('heading', { name: 'A real paper, through the pipeline.' })).toBeVisible();
  return section;
}

function displayNumber(value, precision = 6) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return value == null ? 'Not recorded' : String(value);
  return value.toLocaleString('en-US', { maximumFractionDigits: Number.isInteger(precision) ? Math.max(0, Math.min(12, precision)) : 6 });
}

test('real-paper case study exposes the exact source, question, and finite scope', async ({ page }) => {
  const section = await openPaper(page);
  await expect(section.getByRole('link', { name: receipt.paper.title + ' ↗', exact: true })).toHaveAttribute('href', receipt.paper.url);
  await expect(section.locator('.paper-question')).toHaveText(receipt.question);
  await expect(section.locator('.paper-scope-box')).toContainText(receipt.scope);
  await expect(section.getByText('Finite-horizon evidence, not a proof of asymptotic convergence or a full-paper reproduction.', { exact: true })).toBeVisible();
  const stages = section.getByRole('list', { name: 'Recorded paper pipeline' });
  await expect(stages.locator('li')).toHaveCount(4);
  for (const name of ['Source paper', 'Parse · extract · plan', 'Isolated execution', 'Trusted metrics']) {
    await expect(stages.getByRole('heading', { name, exact: true })).toBeVisible();
  }
  await expect(section.locator('.paper-assurance')).toContainText(receipt.authoring.review);
  for (const limit of receipt.limitations) await expect(section.locator('.paper-assurance')).toContainText(limit);
  await expect(page.getByRole('link', { name: 'Inspect the recorded real-paper run ↘', exact: true })).toHaveAttribute('href', '#real-paper');
  await expect(page.getByRole('status')).toHaveCount(1);
});

test('published metric cards display the recorded trusted values', async ({ page }) => {
  const section = await openPaper(page);
  expect(receipt.metrics.length).toBeGreaterThan(0);
  for (let i = 0; i < receipt.metrics.length; i += 1) {
    const metric = receipt.metrics[i];
    const card = section.getByTestId('paper-metric-' + i);
    await expect(card).toContainText(metric.label);
    await expect(card.locator('strong')).toHaveText(displayNumber(metric.value, metric.precision));
    if (metric.unit) await expect(card).toContainText(metric.unit);
  }
  await expect(section.locator('.paper-chart-caption')).toContainText('the plot does not rerun the experiment');
});

test('recorded trajectory switches metrics and supports keyboard checkpoint inspection', async ({ page }) => {
  const section = await openPaper(page);
  const checkpoints = [...receipt.checkpoints].sort((a, b) => a.step - b.step);
  expect(checkpoints.length).toBeGreaterThan(1);
  const first = checkpoints[0];
  const last = checkpoints[checkpoints.length - 1];
  const position = section.getByRole('button', { name: 'Position', exact: true });
  const regret = section.getByRole('button', { name: 'Average regret', exact: true });
  const slider = section.getByRole('slider', { name: 'Recorded checkpoint' });
  await expect(position).toHaveAttribute('aria-pressed', 'true');
  await expect(section.getByRole('img', { name: /^Recorded Adam and AMSGrad positions/ })).toBeVisible();
  await expect(section.locator('.paper-trace')).toHaveCount(2);
  await expect(slider).toHaveAttribute('aria-valuetext', 'Iteration ' + last.step);
  await slider.focus();
  await slider.press('Home');
  await expect(slider).toHaveValue('0');
  await expect(slider).toHaveAttribute('aria-valuetext', 'Iteration ' + first.step);
  await expect(section.locator('.paper-checkpoint-values')).toContainText(displayNumber(first.adam_position));
  await regret.click();
  await expect(regret).toHaveAttribute('aria-pressed', 'true');
  await expect(position).toHaveAttribute('aria-pressed', 'false');
  await expect(section.getByRole('img', { name: /^Recorded Adam and AMSGrad average regret/ })).toBeVisible();
  await expect(section.locator('.paper-checkpoint-values')).toContainText(displayNumber(first.adam_average_regret));
  await slider.focus();
  await slider.press('End');
  await expect(slider).toHaveAttribute('aria-valuetext', 'Iteration ' + last.step);
  await expect(section.locator('.paper-checkpoint-values')).toContainText(displayNumber(last.amsgrad_average_regret));
});

test('downloaded receipt matches the published case study and exposes the reproduction command', async ({ page }) => {
  const section = await openPaper(page);
  const downloadLink = section.getByRole('link', { name: 'Download recorded run JSON ↓', exact: true });
  await expect(downloadLink).toHaveAttribute('href', 'results/adam-counterexample.json');
  await expect(downloadLink).toHaveAttribute('download', '');
  const [download] = await Promise.all([page.waitForEvent('download'), downloadLink.click()]);
  expect(download.suggestedFilename()).toBe('adam-counterexample.json');
  const downloadedReceipt = JSON.parse(fs.readFileSync(await download.path(), 'utf8'));
  expect(downloadedReceipt).toEqual(receipt);
  await expect(section.getByRole('link', { name: 'Inspect the executable task ↗', exact: true })).toHaveAttribute('href', 'https://github.com/cal2020/Paper2Bench/tree/main/' + receipt.reproduction.source_path);
  await section.locator('.paper-run-details > summary').click();
  await expect(section.locator('.paper-run-details')).toContainText(receipt.paper.pdf_sha256);
  await expect(section.locator('.paper-run-details')).toContainText(receipt.execution.submission_sha256);
  await expect(section.locator('.paper-run-details .code-block')).toContainText(receipt.reproduction.command);
  const copy = section.locator('.paper-run-details .code-block button');
  await copy.click();
  await expect(copy).toHaveText('✓ copied');
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(receipt.reproduction.command);
});
