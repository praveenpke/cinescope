// Smoke-test the running app and capture README screenshots.
// Usage: node scripts/screenshots.mjs  (dev server + API must be running)
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(__dirname, "../../docs/screenshots");
const BASE = process.env.BASE_URL ?? "http://localhost:5173";
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));

async function search(query) {
  await page.fill('input[aria-label="Search movies by description"]', query);
  await page.click('.searchbar button[type="submit"]');
  await page.waitForSelector(".grid .card", { timeout: 15000 });
}

// 1. Landing / empty state
await page.goto(BASE, { waitUntil: "networkidle" });
await page.waitForSelector(".hero h2");
await page.screenshot({ path: `${OUT}/01-landing.png` });
console.log("captured 01-landing.png");

// 2. Search results with interpretation chips
await search("like Inception but funnier");
const cards = await page.locator(".grid .card").count();
if (cards === 0) throw new Error("no result cards rendered");
const chips = await page.locator(".chip").count();
console.log(`results: ${cards} cards, ${chips} interpretation chips`);
await page.screenshot({ path: `${OUT}/02-results.png` });
console.log("captured 02-results.png");

// 3. Detail drawer — verify BOTH more-like-this rows (semantic + behavioral)
await page.locator(".grid .card").first().click();
await page.waitForSelector(".drawer h2", { timeout: 15000 });
await page.waitForSelector(".mlt-row", { timeout: 15000 });
await page.waitForTimeout(400); // let slide-in settle
const rows = await page.locator(".mlt-head h4").count();
console.log(`more-like-this rows: ${rows}`);
await page.screenshot({ path: `${OUT}/03-detail.png` });
console.log("captured 03-detail.png");

// 3b. Scroll drawer to show the behavioral (ALS) row too
const alsHead = page.locator(".mlt-head.als_factors");
if ((await alsHead.count()) > 0) {
  await alsHead.first().scrollIntoViewIfNeeded();
  await page.waitForTimeout(300);
  await page.screenshot({ path: `${OUT}/04-detail-behavioral.png` });
  console.log("captured 04-detail-behavioral.png");
}

if (errors.length) {
  console.error("CONSOLE ERRORS:\n" + errors.join("\n"));
  process.exitCode = 1;
} else {
  console.log("no console errors");
}
await browser.close();
