import { test, expect } from "@playwright/test";

test("Login to practice portal", async ({ page }) => {
  // Given: The user wants to log into the practice portal
  const loginUrl = "https://the-internet.herokuapp.com/login";
  const username = process.env.MCP_TEST_USERNAME || "tomsmith";
  const password = process.env.MCP_TEST_PASSWORD || "SuperSecretPassword!";

  // When: The LLM navigates to the login page and fills the form
  await page.goto(loginUrl);
  await page.getByRole("textbox", { name: "Username" }).fill(username);
  await page.getByRole("textbox", { name: "Password" }).fill(password);
  await page.getByRole("button", { name: /Login/i }).click();

  // Then: The dashboard should be visible
  await expect(page.locator(".flash.success")).toBeVisible();
});
