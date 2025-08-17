from typing import Dict, Any
import pytest
import json
import subprocess
import requests
import time


class TestPlaywrightMCP:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Start the MCP server before tests."""
        self.server_process = subprocess.Popen(
            ["npx", "@playwright/mcp@latest", "--port=8931"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # Wait briefly to ensure the server is up
        time.sleep(2)
        yield
        self.server_process.terminate()

    def test_navigate_and_screenshot(self):
        """Test navigation and screenshot tools."""
        # Given: A running MCP server and a test URL
        url = "https://example.com"
        screenshot_name = "test_screenshot.png"
        expected_response = {"status": "success"}

        # When: Sending navigate and screenshot commands
        navigate_command = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "playwright_navigate",
                "arguments": {"url": url}
            }
        }
        screenshot_command = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "playwright_screenshot",
                "arguments": {"name": screenshot_name}
            }
        }

        # Then: Verify the responses
        result_navigate = self._send_command(navigate_command)
        result_screenshot = self._send_command(screenshot_command)

        assert result_navigate == expected_response, "Navigation failed"
        assert result_screenshot == expected_response, "Screenshot failed"

    def test_fill_and_click(self):
        """Test form filling and clicking tools."""
        # Given: A running MCP server and a test login page
        url = "https://example.com/login"
        username_selector = "#username"
        password_selector = "#password"
        login_button = "#loginButton"
        username = "testuser"
        password = "testpass"
        expected_response = {"status": "success"}

        # When: Sending navigate, fill, and click commands
        navigate_command = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "playwright_navigate",
                "arguments": {"url": url}
            }
        }
        fill_username_command = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "playwright_fill",
                "arguments": {"selector": username_selector, "value": username}
            }
        }
        fill_password_command = {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "playwright_fill",
                "arguments": {"selector": password_selector, "value": password}
            }
        }
        click_command = {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "playwright_click",
                "arguments": {"selector": login_button}
            }
        }

        # Then: Verify the responses
        result_navigate = self._send_command(navigate_command)
        result_fill_username = self._send_command(fill_username_command)
        result_fill_password = self._send_command(fill_password_command)
        result_click = self._send_command(click_command)

        assert result_navigate == expected_response, "Navigation failed"
        assert result_fill_username == expected_response, "Username fill failed"
        assert result_fill_password == expected_response, "Password fill failed"
        assert result_click == expected_response, "Click failed"

    def test_api_post(self):
        """Test API POST request tool."""
        # Given: A running MCP server and a test API endpoint
        api_url = "https://api.example.com/login"
        request_body = {"username": "testuser", "password": "testpass"}
        headers = {"Content-Type": "application/json"}
        expected_response = {"status": "success"}

        # When: Sending an API POST command
        post_command = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "api_post",
                "arguments": {
                    "url": api_url,
                    "body": request_body,
                    "headers": headers
                }
            }
        }

        # Then: Verify the response
        result_post = self._send_command(post_command)

        assert result_post == expected_response, "API POST failed"

    def _send_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send JSON-RPC command to the MCP server and return the response."""
        try:
            response = requests.post(
                "http://localhost:8931/mcp",
                json=command,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Failed to send command: {str(e)}")
