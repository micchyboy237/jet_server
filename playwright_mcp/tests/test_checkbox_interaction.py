import pytest
from playwright.async_api import async_playwright


@pytest.mark.asyncio
async def test_check_uncheck_checkbox_1():
    # Given: The user wants to interact with checkboxes
    url = "https://the-internet.herokuapp.com/checkboxes"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        # When: Navigate to the checkbox page
        await page.goto(url, timeout=60000)
        checkbox1 = page.get_by_role("checkbox").first
        print(f"Page URL after navigation: {page.url}")
        print(f"Checkbox 1 locator resolved: {checkbox1}")
        await checkbox1.is_visible(timeout=10000)
        print(f"Checkbox 1 visible: {await checkbox1.is_visible()}")

        # When: Check and uncheck first checkbox
        result_initial_state = await checkbox1.is_checked()
        await checkbox1.check()
        result_checked_state = await checkbox1.is_checked()
        await checkbox1.uncheck()
        result_unchecked_state = await checkbox1.is_checked()

        # Then: Verify state transitions
        expected_initial_state = False
        expected_checked_state = True
        expected_unchecked_state = False
        assert result_initial_state == expected_initial_state, "First checkbox should not be checked initially"
        assert result_checked_state == expected_checked_state, "First checkbox should be checked after checking"
        assert result_unchecked_state == expected_unchecked_state, "First checkbox should not be checked after unchecking"

        await browser.close()
