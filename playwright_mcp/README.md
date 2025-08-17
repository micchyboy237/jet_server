# Playwright MCP Example Project

This project demonstrates how to use Playwright MCP (Model Context Protocol) with a Large Language Model (LLM) to automate browser interactions, such as navigating URLs, logging into a web application, and interacting with checkboxes. It includes both TypeScript tests (using Playwright’s test runner) and Python tests (using pytest). The project is optimized for a Mac M1 environment, runs in non-headless mode to make the browser visible, and follows best practices for modularity, testability, and clarity.

## Prerequisites

- **Node.js (18+)**: Install via `brew install node`.
- **Python (3.8+)**: Install via `brew install python`.
- **Git** (optional): Install via `brew install git`.
- **Playwright MCP**: Installed as a Node.js dependency.
- **pytest**: Installed for Python tests.

## Setup Instructions

1. **Clone or Create the Project Directory**:

   ```bash
   mkdir playwright-mcp-example
   cd playwright-mcp-example
   ```

2. **Initialize Node.js Project**:

   ```bash
   npm init -y
   ```

3. **Install Dependencies**:

   ```bash
   npm install -D @playwright/test @playwright/mcp dotenv
   npx playwright install
   pip3 install pytest pytest-playwright pytest-asyncio
   ```

4. **Set Up Environment Variables**:
   Create a `.env` file in the project root:

   ```env
   MCP_TEST_USERNAME=tomsmith
   MCP_TEST_PASSWORD=SuperSecretPassword!
   ```

   > **Note**: These credentials are for the login test on `https://the-internet.herokuapp.com/login`. Update if using a different login URL.

5. **Create Playwright Configuration**:
   Save the following as `playwright.config.ts`:

   ```typescript
   import { defineConfig } from "@playwright/test";
   export default defineConfig({
     use: {
       browserName: "chromium",
       headless: false,
     },
     timeout: 30000,
   });
   ```

6. **Create MCP Configuration**:
   Save the following as `mcp-config.json`:

   ```json
   {
     "browser": {
       "browserName": "chromium",
       "isolated": true,
       "launchOptions": {
         "headless": false
       },
       "contextOptions": {
         "viewport": { "width": 1280, "height": 720 }
       }
     },
     "server": {
       "port": 3000,
       "host": "localhost"
     },
     "capabilities": ["tabs", "pdf"],
     "outputDir": "./mcp-output",
     "network": {
       "allowedOrigins": [
         "https://the-internet.herokuapp.com",
         "https://practiceautomatedtesting.com"
       ],
       "blockedOrigins": []
     },
     "imageResponses": "omit"
   }
   ```

7. **Organize Tests**:
   - Place TypeScript tests in `tests-examples/`:
     - `title-verification.spec.ts`
     - `mcp-login.spec.ts`
     - `checkbox-interaction.spec.ts`
   - Place Python tests in `tests/`:
     - `test_title_verification.py`
     - `test_mcp_login.py`
     - `test_checkbox_interaction.py`

## Running Tests (Non-Headless Mode)

All tests are configured to run in **non-headless mode**, so the Chromium browser will be visible during execution, showing actions like navigating URLs, filling forms, or clicking checkboxes. Tests use `https://the-internet.herokuapp.com` for title verification and login, and `https://practiceautomatedtesting.com/webelements` for checkbox interactions.

### TypeScript Tests (Playwright)

1. **Start the Playwright MCP Server**:

   ```bash
   npx @playwright/mcp@latest --config mcp-config.json
   ```

   Keep this running in a separate terminal. The browser will open visibly when tests run.

2. **Run All Tests**:

   ```bash
   npx playwright test tests-examples/
   ```

   The Chromium browser will open, showing test actions (e.g., navigating to `https://the-internet.herokuapp.com`).

3. **Run a Specific Test**:

   ```bash
   npx playwright test tests-examples/title-verification.spec.ts
   ```

4. **Run with UI Mode (Debugging)**:
   ```bash
   npx playwright test tests-examples/ --ui
   ```
   This opens Playwright’s UI mode with a visible browser for interactive debugging.

### Python Tests (pytest)

1. **Run All Tests**:

   ```bash
   pytest tests/
   ```

   The browser will open visibly, showing test actions. All test files are configured with `headless=False`.

2. **Run a Specific Test**:

   ```bash
   pytest tests/test_title_verification.py
   ```

3. **Run with Verbose Output**:
   ```bash
   pytest tests/ -v
   ```

## Test Descriptions

- **Title Verification**: Navigates to `https://the-internet.herokuapp.com` and verifies the page title starts with “The Internet”.
- **Login Test**: Logs into `https://the-internet.herokuapp.com/login` using credentials `tomsmith` and `SuperSecretPassword!`, verifying the success message.
- **Checkbox Interaction**: Interacts with checkboxes on `https://practiceautomatedtesting.com/webelements` and validates their state. If this URL fails, use `https://the-internet.herokuapp.com/checkboxes` as a fallback (update test files accordingly).

## Notes

- **Mac M1 Compatibility**: All tools are compatible with Apple Silicon. No additional configuration is needed.
- **URLs**: Tests use `https://the-internet.herokuapp.com` for title and login tests, and `https://practiceautomatedtesting.com/webelements` for checkbox tests. If the checkbox URL fails, replace it with `https://the-internet.herokuapp.com/checkboxes` in `checkbox-interaction.spec.ts` and `test_checkbox_interaction.py`.
- **Non-Headless Mode**: All configurations (`mcp-config.json`, `playwright.config.ts`, and Python tests) use `headless: false` to ensure the browser is visible.
- **Debugging**: Use `--ui` for Playwright tests or `-v` for pytest to diagnose issues. Check debug logs in `checkbox-interaction.spec.ts` and `test_checkbox_interaction.py` for checkbox test failures. If the browser doesn’t appear or URLs fail, share error outputs.
- **Dependencies**: Uses modern, free packages (`@playwright/test`, `@playwright/mcp`, `pytest`, `pytest-playwright`, `pytest-asyncio`).

## Troubleshooting

- **Browser Not Visible**: Ensure `headless: false` in `mcp-config.json`, `playwright.config.ts`, and Python tests. Verify the MCP server is running (`npx @playwright/mcp@latest --config mcp-config.json`).
- **URL Not Loading**: Verify internet connectivity or try alternative URLs (e.g., `https://the-internet.herokuapp.com/checkboxes` for checkbox tests). Check debug logs in checkbox tests for page URL and element visibility.
- **Checkbox Test Failure**: If `https://practiceautomatedtesting.com/webelements` fails, use the alternative test versions with `https://the-internet.herokuapp.com/checkboxes`. Check debug logs for visibility issues.
- **Node.js Issues**: Reinstall with `brew reinstall node`.
- **Browser Issues**: Run `npx playwright install` to ensure browsers are installed.
- **MCP Server**: Ensure `npx @playwright/mcp@latest --config mcp-config.json` is running before tests.
- **Python Issues**: Use `python3` and `pip3` to avoid version conflicts.
- **Port Conflicts**: If port 3000 is in use, update `mcp-config.json` to use a different port (e.g., 3001).

For further assistance, contact the project maintainer or provide error logs for specific issues.
