import { test, expect } from "@playwright/test";

test("Verify page title on navigation", async ({ page }) => {
  // Given: The user wants to navigate to a specific URL
  const url = "https://the-internet.herokuapp.com";
  const expectedTitlePrefix = "The Internet";

  // When: The LLM navigates to the URL and retrieves the title
  await page.goto(url);
  const title = await page.title();

  // Then: The title should start with the expected prefix
  const isTitleValid = title.startsWith(expectedTitlePrefix);
  await expect(isTitleValid).toBe(true);
});
