SCRIPT_DIR="${0:A:h}"
MCP_CONFIG="${SCRIPT_DIR}/mcp-config.json"

npx @playwright/mcp@latest --config "$MCP_CONFIG"
