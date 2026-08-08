import { defineConfig } from '@playwright/test';
export default defineConfig({
  testDir: './tests',
  use: {
    baseURL: 'http://127.0.0.1:4321/sketchsense/',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'node scripts/serve-preview.mjs',
    port: 4321,
    reuseExistingServer: true,
  },
  projects: [{ name: 'chromium', use: { browserName: 'chromium' } }],
});
