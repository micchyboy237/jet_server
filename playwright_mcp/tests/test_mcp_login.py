import pytest
from playwright.async_api import async_playwright
import os


@pytest.mark.asyncio
async def test_login_to_practice_portal():
    # Given: The user wants to log into the practice portal
    login_url = "https://the-internet.herokuapp.com/login"
    expected_username = os.getenv("MCP_TEST_USERNAME", "tomsmith")
    expected_password = os.getenv("MCP_TEST_PASSWORD", "SuperSecretPassword!")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        # When: Navigate to the login page and fill the form
        await page.goto(login_url)
        await page.get_by_role("textbox", name="Username").fill(expected_username)
        await page.get_by_role("textbox", name="Password").fill(expected_password)
        await page.get_by_role("button", name="Login", exact=False).click()

        # Then: Verify the dashboard is visible
        result_dashboard = await page.locator(".flash.success").is_visible()
        expected_is_visible = True
        assert result_dashboard == expected_is_visible, "Expected success message to be visible after login"

        await browser.close()
