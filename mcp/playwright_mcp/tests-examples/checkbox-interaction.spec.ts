import { test, expect } from "@playwright/test";

test.describe("Checkbox Section - the-internet.herokuapp.com", () => {
  test.beforeEach(async ({ page }) => {
    // Given: The user wants to interact with checkboxes
    await page.goto("https://the-internet.herokuapp.com/checkboxes", {
      timeout: 60000,
    });
    await expect(page.getByRole("checkbox").first()).toBeVisible({
      timeout: 10000,
    });
    console.log("Page URL after navigation:", await page.url());
    console.log(
      "First checkbox visible:",
      await page.getByRole("checkbox").first().isVisible()
    );
  });

  test("Check and uncheck first checkbox", async ({ page }) => {
    // When: The LLM checks and unchecks the first checkbox
    const checkbox1 = page.getByRole("checkbox").first();
    await expect(checkbox1).toBeVisible({ timeout: 10000 });
    console.log(
      "Checkbox 1 resolved and visible:",
      await checkbox1.isVisible()
    );
    await expect(checkbox1).not.toBeChecked();
    await checkbox1.check();
    await expect(checkbox1).toBeChecked();
    await checkbox1.uncheck();
    await expect(checkbox1).not.toBeChecked();
  });
});
