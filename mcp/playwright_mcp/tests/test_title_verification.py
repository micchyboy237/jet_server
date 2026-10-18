import pytest
from playwright.async_api import async_playwright


@pytest.mark.asyncio
async def test_verify_page_title():
    # Given: The user wants to navigate to a specific URL
    url = "https://the-internet.herokuapp.com"
    expected_title_prefix = "The Internet"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        # When: Navigate to the URL and retrieve the title
        await page.goto(url)
        result_title = await page.title()

        # Then: Verify the title starts with the expected prefix
        expected_is_valid = result_title.startswith(expected_title_prefix)
        assert expected_is_valid, f"Expected title to start with '{expected_title_prefix}', got '{result_title}'"

        await browser.close()
