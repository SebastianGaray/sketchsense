import { expect, test } from '@playwright/test';

test('loads, draws, predicts, clears, and keeps navigation working', async ({
  page,
}) => {
  const requests: string[] = [];
  page.on('request', (request) => requests.push(request.url()));
  await page.goto('en/');
  await expect(page).toHaveTitle(/SketchSense/);
  await expect(page.locator('body')).not.toContainText(/[\u00c3\u00c2]/);
  await expect(page.locator('[data-status]')).toContainText('Model ready', {
    timeout: 30_000,
  });
  await expect(page.locator('[data-global-control="theme"]')).toBeVisible();
  await expect(page.locator('[data-global-control="language"]')).toBeVisible();
  await expect(page.locator('.page-nav')).toBeVisible();
  await expect(page.locator('.page-nav a')).toHaveCount(5);
  const canvas = page.locator('[data-canvas]');
  await canvas.scrollIntoViewIfNeeded();
  await expect(page.locator('[data-stroke-width]')).toHaveValue('14');
  await page.locator('[data-stroke-width]').fill('10');
  await expect(page.locator('[data-stroke-output]')).toHaveText('10 px');
  const box = await canvas.boundingBox();
  if (!box) throw new Error('Canvas missing');
  await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.7);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.25, {
    steps: 8,
  });
  await expect(page.locator('[data-results] li')).toHaveCount(3, {
    timeout: 30_000,
  });
  await page.mouse.move(box.x + box.width * 0.7, box.y + box.height * 0.7, {
    steps: 8,
  });
  await page.mouse.up();
  await expect(page.locator('[data-results] li')).toHaveCount(3, {
    timeout: 30_000,
  });
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
  await page.getByRole('radio', { name: '28 × 28 pixels' }).check();
  await expect(canvas).toHaveAttribute('data-input-mode', 'pixels');
  await expect(page.locator('[data-mode-help]')).toContainText(
    'directly on the enlarged 28 × 28',
  );
  await canvas.scrollIntoViewIfNeeded();
  const pixelBox = await canvas.boundingBox();
  if (!pixelBox) throw new Error('Pixel canvas missing');
  await page.mouse.move(
    pixelBox.x + pixelBox.width * 0.2,
    pixelBox.y + pixelBox.height * 0.5,
  );
  await page.mouse.down();
  await page.mouse.move(
    pixelBox.x + pixelBox.width * 0.8,
    pixelBox.y + pixelBox.height * 0.5,
    { steps: 20 },
  );
  await page.mouse.up();
  await expect(page.locator('[data-results] li')).toHaveCount(3, {
    timeout: 30_000,
  });
  await page.locator('[data-clear]').click();
  await expect(page.locator('[data-predict]')).toBeDisabled();
  await page.locator('[data-theme-control] summary').click();
  await page.getByRole('button', { name: 'Dark' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await expect(page.locator('[data-theme-control]')).not.toHaveAttribute(
    'open',
    '',
  );
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.getByRole('link', { name: 'Espa\u00f1ol' }).click();
  await expect(page).toHaveURL(/\/es\/$/);
  await expect(page.getByRole('heading', { level: 1 })).toHaveText(
    /rea de dibujo$/,
  );
});

test('lists every supported category with validated held-out prompts', async ({
  page,
}) => {
  await page.goto('en/examples/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText(
    'What can SketchSense recognize?',
  );
  await expect(page.locator('.category-grid li')).toHaveCount(16);
  await expect(page.locator('.category-icon')).toHaveCount(16);
  await expect(
    page.locator('.category-grid li').filter({ hasText: 'cat' }).first(),
  ).toBeVisible();
  await expect(page.getByText('held-out Quick, Draw! sketch')).toBeVisible();
  await expect(page.locator('.category-grid img')).toHaveCount(16);
  await expect(page.locator('.category-grid img').first()).toHaveAttribute(
    'src',
    /examples\/v3\/.+\.png/,
  );
  await expect(page.locator('.category-grid img').last()).toHaveAttribute(
    'src',
    /examples\/v3\/.+\.png/,
  );
  await expect(page.getByRole('link', { name: 'Examples' })).toHaveAttribute(
    'aria-current',
    'page',
  );
  await page.getByRole('link', { name: 'Canvas', exact: true }).click();
  await expect(page).toHaveURL(/\/en\/$/);
  await page.getByRole('link', { name: 'Model' }).click();
  await expect(page).toHaveURL(/\/en\/model\/$/);
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('Model');
  await page.getByRole('link', { name: 'About' }).click();
  await expect(page).toHaveURL(/\/en\/about\/$/);
  await expect(page.getByRole('heading', { level: 1 })).toContainText(
    'From stroke to prediction',
  );
  await page.getByRole('link', { name: 'Engineering' }).click();
  await expect(page).toHaveURL(/\/en\/engineering\/$/);
  await expect(page.getByRole('heading', { level: 1 })).toContainText(
    'specified, assisted, and verified',
  );
  await expect(page.locator('.engineering-grid li')).toHaveCount(4);
  await page.getByRole('link', { name: 'Canvas', exact: true }).click();
  await expect(page).toHaveURL(/\/en\/$/);
});

test('fits a mobile viewport without horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 800 });
  await page.goto('en/');
  await page.locator('[data-theme-control] summary').click();
  const themeMenu = await page.locator('.theme-menu').boundingBox();
  if (!themeMenu) throw new Error('Theme menu missing');
  expect(themeMenu.x).toBeGreaterThanOrEqual(0);
  expect(themeMenu.x + themeMenu.width).toBeLessThanOrEqual(320);
  expect(
    await page.evaluate(
      () =>
        document.documentElement.scrollWidth <=
        document.documentElement.clientWidth,
    ),
  ).toBe(true);
});
