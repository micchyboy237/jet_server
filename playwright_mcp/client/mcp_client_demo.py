import requests

BASE_URL = "http://localhost:8000"


def list_servers():
    """Fetch and print all connected MCP servers."""
    try:
        resp = requests.get(f"{BASE_URL}/servers")
        resp.raise_for_status()
        servers = resp.json()
        if "error" in servers:
            print(f"Error fetching servers: {servers['error']}")
        else:
            print("Servers:", servers)
    except requests.RequestException as e:
        print(f"Failed to fetch servers: {str(e)}")


def list_tools(server_name: str):
    """List all tools for a given MCP server."""
    try:
        resp = requests.get(f"{BASE_URL}/tools/{server_name}")
        resp.raise_for_status()
        print(f"Tools on {server_name}:", resp.json())
    except requests.RequestException as e:
        print(f"Failed to list tools for {server_name}: {str(e)}")


def call_tool(server_name: str, tool_name: str, args: dict):
    """Call a tool on the MCP server."""
    try:
        resp = requests.post(
            f"{BASE_URL}/tools/{server_name}/{tool_name}",
            json={"args": args},
        )
        resp.raise_for_status()
        print(f"Result of {tool_name}:", resp.json())
    except requests.RequestException as e:
        print(f"Failed to call {tool_name} on {server_name}: {str(e)}")


if __name__ == "__main__":
    list_servers()
    list_tools("playwright")
    call_tool("playwright", "browser_navigate", {"url": "https://example.com"})
