import { chromium } from 'playwright';
import { mkdirSync } from 'node:fs';
import path from 'node:path';

const BASE_URL = process.argv[2] ?? 'http://localhost:5199';
const OUT_DIR = process.argv[3] ?? 'parity-baseline';

const VIEWPORTS = [
  { w: 390, h: 844 },
  { w: 768, h: 1024 },
  { w: 1024, h: 768 },
  { w: 1440, h: 900 },
];

// [routeSlug, routePath, requiresAuth]
const ROUTES = [
  ['login', '/login', false],
  ['home', '/', true],
  ['gw-overview', '/gw-overview', true],
  ['command-center', '/command-center', true],
  ['players', '/players', true],
  ['player-detail', '/players/302', true],
  ['fixtures', '/fixtures', true],
];

mkdirSync(OUT_DIR, { recursive: true });

const browser = await chromium.launch();
let count = 0;

for (const vp of VIEWPORTS) {
  const context = await browser.newContext({
    viewport: { width: vp.w, height: vp.h },
    deviceScaleFactor: 1,
  });
  for (const [slug, route, needsAuth] of ROUTES) {
    if (needsAuth) {
      await context.addInitScript(() => {
        window.localStorage.setItem('teamId', '12345');
      });
    }
    const page = await context.newPage();
    try {
      await page.goto(`${BASE_URL}${route}`, { waitUntil: 'networkidle', timeout: 30000 });
    } catch {
      // networkidle can time out on long-polling pages; proceed with what loaded
    }
    await page.waitForTimeout(1200);
    await page.screenshot({
      path: path.join(OUT_DIR, `${slug}-${vp.w}.png`),
      fullPage: true,
    });
    await page.close();
    count += 1;
    console.log(`captured ${slug}-${vp.w}.png`);
  }
  await context.close();
}

await browser.close();
console.log(`done: ${count} screenshots -> ${OUT_DIR}`);
