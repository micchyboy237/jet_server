import asyncio
import json
import re
from typing import List, Dict, Any
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from pydantic import BaseModel, ValidationError, validate_call
from jet.logger import CustomLogger
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

LOGS_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_server/playwright_mcp/server/sample2/logs"
logger = CustomLogger(f"{LOGS_DIR}/mcp_agent.log", overwrite=True)
MCP_SERVER_PATH = "mcp_server.py"
MODEL_PATH = "mlx-community/Qwen3-1.7B-4bit-DWQ-053125"


class ToolRequest(BaseModel):
    tool: str
    arguments: Dict[str, Any]


async def discover_tools() -> List[Dict]:
    server_params = StdioServerParameters(
        command="python", args=[MCP_SERVER_PATH])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [{"name": t.name, "description": t.description, "schema": t.inputSchema, "outputSchema": t.outputSchema} for tool_type, tool_list in tools if tool_type == "tools" for t in tool_list]


async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    server_params = StdioServerParameters(
        command="python", args=[MCP_SERVER_PATH])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            for attempt in range(3):
                try:
                    await session.initialize()
                    logger.debug(
                        f"Calling tool: {tool_name} with arguments: {arguments}")
                    # Wrap arguments in a nested 'arguments' field
                    wrapped_arguments = {"arguments": arguments}
                    result = await session.call_tool(tool_name, wrapped_arguments)
                    return str(result[0]["content"] if isinstance(result, list) and result and "content" in result[0] else result)
                except Exception as e:
                    if attempt == 2:
                        logger.error(
                            f"Tool execution failed after 3 attempts: {str(e)}")
                        return f"Error executing {tool_name}: {str(e)}"
                    await asyncio.sleep(0.5)


async def query_llm(prompt: str, tool_info: List[Dict], previous_messages: List[Dict] = []) -> tuple[str, List[Dict]]:
    tool_descriptions = "\n\n".join(
        [f"Tool: {t['name']}\nDescription: {t['description']}\nInput Schema: {json.dumps(t['schema'], indent=2)}\nOutput Schema: {json.dumps(t['outputSchema'], indent=2)}" for t in tool_info])
    system_prompt = f"You are an AI assistant with MCP tools:\n{tool_descriptions}\nUse JSON for tool requests: {{'tool': 'name', 'arguments': {{'arg': 'value'}}}}."
    messages = [m for m in previous_messages if m["role"]
                != "system"] + [{"role": "user", "content": prompt}]

    # Format messages for MLX
    formatted_messages = [
        {"role": "system", "content": system_prompt}] + messages

    try:
        # Load MLX model
        model, tokenizer = load(MODEL_PATH)
        sampler = make_sampler(temp=0.7)

        # Generate response
        llm_response = generate(
            model,
            tokenizer,
            prompt=tokenizer.apply_chat_template(
                formatted_messages, tokenize=False, enable_thinking=False),
            max_tokens=4000,
            sampler=sampler,
            verbose=True,
        )

        try:
            json_match = re.search(r'(\{[\s\S]*\})', llm_response)
            if json_match:
                tool_request = ToolRequest.model_validate_json(
                    json_match.group(1))
                logger.debug(f"Tool request: {tool_request}")
                for tool in tool_info:
                    if tool["name"] == tool_request.tool:
                        try:
                            # Validate arguments against the tool's input schema
                            @validate_call(config=dict(validate_json_schema=True))
                            def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> None:
                                pass
                            validate_schema(
                                tool_request.arguments, tool["schema"])
                        except ValidationError as e:
                            return f"Invalid tool arguments: {str(e)}", messages
                        tool_result = await execute_tool(tool_request.tool, tool_request.arguments)
                        messages.append(
                            {"role": "assistant", "content": llm_response})
                        messages.append(
                            {"role": "user", "content": f"Tool result: {tool_result}"})
                        # Generate follow-up response with tool result
                        follow_up_messages = [
                            {"role": "system", "content": system_prompt}] + messages
                        llm_response = generate(
                            model,
                            tokenizer,
                            prompt=tokenizer.apply_chat_template(
                                follow_up_messages, tokenize=False, enable_thinking=False),
                            max_tokens=4000,
                            sampler=sampler,
                            verbose=True,
                        )
                        return llm_response, messages
        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            logger.debug(f"No valid tool request detected: {str(e)}")
        return llm_response, messages
    except Exception as e:
        logger.error(f"MLX inference failed: {str(e)}")
        return f"Error querying LLM: {str(e)}", messages


async def chat_session():
    tools = await discover_tools()
    logger.debug(
        f"Discovered {len(tools)} tools: {[t['name'] for t in tools]}")
    messages = []
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            logger.debug("Ending chat session.")
            break
        response, messages = await query_llm(user_input, tools, messages)
        print(f"Assistant: {response}")

if __name__ == "__main__":
    asyncio.run(chat_session())
