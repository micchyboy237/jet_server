import http from "http";

import { createConnection } from "@playwright/mcp";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";

http.createServer(async (req, res) => {
  // ...

  // Creates a headless Playwright MCP server with SSE transport
  const connection = await createConnection({
    browser: { launchOptions: { headless: true } },
  });
  const transport = new SSEServerTransport("/messages", res);
  await connection.sever.connect(transport);

  // ...
});
