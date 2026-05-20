#!/usr/bin/env npx tsx
import { chromium } from "@playwright/test";
import { mkdirSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const mediaDir = resolve(__dirname, "../media");
mkdirSync(mediaDir, { recursive: true });

const BASE_URL = "http://localhost:3000";

const ROLES = [
  { key: "soc_tier1", label: "SOC Tier 1 Analyst" },
  { key: "soc_tier2", label: "SOC Tier 2 Analyst" },
  { key: "sre", label: "SRE Engineer" },
  { key: "contractor", label: "Contractor" },
  { key: "ai_agent", label: "AI Agent" },
];

async function main() {
  console.log("Launching browser...");
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    colorScheme: "dark",
  });

  const page = await context.newPage();
  await page.goto(BASE_URL, { waitUntil: "networkidle" });
  await page.waitForTimeout(2000);

  // Screenshot 1: Default dashboard (SOC Tier 1)
  console.log("Taking dashboard screenshot...");
  await page.screenshot({
    path: resolve(mediaDir, "01-dashboard-overview.png"),
    fullPage: false,
  });

  // Screenshot each role
  for (let i = 0; i < ROLES.length; i++) {
    const role = ROLES[i];
    console.log(`Switching to ${role.label}...`);

    // Click the role selector dropdown
    const trigger = page.locator('[role="combobox"]').first();
    await trigger.click();
    await page.waitForTimeout(500);

    // Select the role
    const option = page.getByText(role.label, { exact: false }).first();
    try {
      await option.click({ timeout: 3000 });
    } catch {
      // Try alternative selector
      const options = page.locator('[role="option"]');
      const count = await options.count();
      for (let j = 0; j < count; j++) {
        const text = await options.nth(j).textContent();
        if (text?.includes(role.label) || text?.includes(role.key)) {
          await options.nth(j).click();
          break;
        }
      }
    }
    await page.waitForTimeout(1500);

    // Click first incident if available
    const incidents = page.locator("aside button, aside [role='button'], aside .cursor-pointer").first();
    try {
      await incidents.click({ timeout: 2000 });
      await page.waitForTimeout(1000);
    } catch {
      // No incidents clickable
    }

    await page.screenshot({
      path: resolve(mediaDir, `0${i + 2}-role-${role.key}.png`),
      fullPage: false,
    });
    console.log(`  Saved 0${i + 2}-role-${role.key}.png`);
  }

  // Screenshot the AuthZ Log tab
  console.log("Taking AuthZ Log screenshot...");
  const authzTab = page.getByText("AuthZ Log", { exact: false }).first();
  try {
    await authzTab.click();
    await page.waitForTimeout(1000);
    await page.screenshot({
      path: resolve(mediaDir, "07-authz-log.png"),
      fullPage: false,
    });
  } catch {
    console.log("  Could not click AuthZ Log tab");
  }

  // Screenshot the SPL Query tab
  console.log("Taking SPL Query screenshot...");
  const queryTab = page.getByText("SPL Query", { exact: false }).first();
  try {
    await queryTab.click();
    await page.waitForTimeout(1000);
    await page.screenshot({
      path: resolve(mediaDir, "08-spl-query.png"),
      fullPage: false,
    });
  } catch {
    console.log("  Could not click SPL Query tab");
  }

  await browser.close();
  console.log(`\nDone! Screenshots saved to ${mediaDir}`);
}

main().catch((err) => {
  console.error("Screenshot script failed:", err);
  process.exit(1);
});
