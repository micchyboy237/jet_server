curl -X POST http://localhost:8931/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "0.1.0",
      "clientInfo": {
        "name": "curl-client",
        "version": "0.0.1"
      },
      "capabilities": {}
    }
  }'
