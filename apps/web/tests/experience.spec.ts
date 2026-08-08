import { expect, test } from '@playwright/test';

test('loads, draws, predicts, clears, and keeps navigation working', async ({
  page,
}) => {
  const requests: string[] = [];
  page.on('request', (request) => requests.push(request.url()));
  await page.goto('en/');
  await expect(page).toHaveTitle(/SketchSense/);
  await expect(page.locator('[data-status]')).toContainText('Model ready', {
    timeout: 30_000,
  });
  const canvas = page.locator('[data-canvas]');
  await canvas.scrollIntoViewIfNeeded();
  const box = await canvas.boundingBox();
  if (!box) throw new Error('Canvas missing');
  await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.7);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.25, {
    steps: 8,
  });
  await page.mouse.move(box.x + box.width * 0.7, box.y + box.height * 0.7, {
    steps: 8,
  });
  await page.mouse.up();
  await page.locator('[data-predict]').click();
  await expect(page.locator('[data-results] li')).toHaveCount(3, {
    timeout: 30_000,
  });
  expect(
    requests.every(
      (url) => !/upload|predict|infer/i.test(new URL(url).pathname),
    ),
  ).toBe(true);
  await page.locator('[data-clear]').click();
  await expect(page.locator('[data-predict]')).toBeDisabled();
  await page.getByRole('button', { name: 'Dark' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.getByRole('link', { name: 'Español' }).click();
  await expect(page).toHaveURL(/\/es\/$/);
  await expect(page.getByRole('heading', { level: 1 })).toHaveText(
    /rea de dibujo$/,
  );
});

test('fits a mobile viewport without horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 800 });
  await page.goto('en/');
  expect(
    await page.evaluate(
      () =>
        document.documentElement.scrollWidth <=
        document.documentElement.clientWidth,
    ),
  ).toBe(true);
});
