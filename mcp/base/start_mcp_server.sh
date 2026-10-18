SCRIPT_DIR="${0:A:h}"
MCP_CONFIG="${SCRIPT_DIR}/mcp-config.json"

# Extract outputDir from the config file
OUTPUT_DIR=$(grep -o '"outputDir": *"[^"]*"' "$MCP_CONFIG" | awk -F'"' '{print $4}')

if [ -n "$OUTPUT_DIR" ]; then
  echo "Cleaning output directory: $OUTPUT_DIR"
  # Create directory if it doesn't exist
  mkdir -p "$OUTPUT_DIR"
  # Remove all files and subdirectories inside
  rm -rf "${OUTPUT_DIR:?}"/*
else
  echo "Warning: Could not find outputDir in $MCP_CONFIG"
fi

echo "Starting Playwright MCP server..."
npx @playwright/mcp@latest --config "$MCP_CONFIG"