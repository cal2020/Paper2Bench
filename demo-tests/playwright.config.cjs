const { defineConfig, devices } = require('@playwright/test');

const port = Number(process.env.P2B_DEMO_TEST_PORT || '4173');
if (!Number.isInteger(port) || port < 1024 || port > 65535) {
  throw new Error('P2B_DEMO_TEST_PORT must be an integer from 1024 to 65535');
}

module.exports = defineConfig({
  testDir: './specs',
  forbidOnly: Boolean(process.env.CI),
  fullyParallel: true,
  workers: process.env.CI ? 2 : undefined,
  retries: 0,
  timeout: 30_000,
  expect: { timeout: 8_000 },
  outputDir: './test-results',
  reporter: [
    ['line'],
    ['html', { outputFolder: 'playwright-report', open: 'never' }],
    ['junit', { outputFile: 'test-results/junit.xml' }],
  ],
  use: {
    launchOptions: process.env.P2B_CHROMIUM_EXECUTABLE
      ? { executablePath: process.env.P2B_CHROMIUM_EXECUTABLE }
      : {},
    baseURL: `http://127.0.0.1:${port}`,
    serviceWorkers: 'block',
    reducedMotion: 'reduce',
    permissions: ['clipboard-read', 'clipboard-write'],
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
    video: 'off',
  },
  projects: [
    { name: 'chromium-desktop', use: { ...devices['Desktop Chrome'] } },
    { name: 'chromium-mobile', use: { ...devices['Pixel 7'] } },
  ],
  webServer: {
    command: 'node serve.cjs',
    cwd: __dirname,
    url: `http://127.0.0.1:${port}/index.html`,
    env: { P2B_DEMO_TEST_PORT: String(port) },
    reuseExistingServer: false,
    timeout: 10_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
});
