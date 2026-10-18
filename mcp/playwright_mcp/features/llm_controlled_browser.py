from playwright.async_api import async_playwright, Page
from typing import List, Dict, Any
import json

from jet.llm.mlx.generation import generate
from jet.logger import logger
from jet.models.model_registry.transformers.mlx_model_registry import MLXModelRegistry
from jet.models.model_types import LLMModelType
from jet.utils.markdown import extract_json_block_content


class LLMCommandParser:
    """Parses natural language commands using a local MLX LLM and executes browser actions."""

    def __init__(self, page: Page, model: LLMModelType = "qwen3-1.7b-4bit"):
        self.page = page
        self.model = MLXModelRegistry.load_model(model)
        self.action_handlers = {
            "navigate": self._navigate,
            "click": self._click,
            "fill": self._fill,
            "extract_text": self._extract_text
        }

    async def _call_llm(self, instruction: str) -> Dict[str, Any]:
        """Calls the local MLX LLM to parse the instruction."""
        try:
            prompt = (
                f"Parse this browser automation command into a JSON object with 'action' (one of: navigate, click, fill, extract_text) "
                f"and 'parameters' (list of values). Example: For 'navigate to https://example.com', return "
                f"{{'action': 'navigate', 'parameters': ['https://example.com']}}. For 'fill textbox named Username with tomsmith', "
                f"return {{'action': 'fill', 'parameters': ['textbox named Username', 'tomsmith']}}. "
                f"Command: '{instruction}'"
            )
            response_stream = self.model.stream_generate(
                prompt,
                max_tokens=100,
                temperature=0.5,
                verbose=True
            )
            content = ""
            for response in response_stream:
                chunk = response["choices"][0]["text"]
                content += chunk

            # Assume response is a JSON string; parse it
            try:
                parsed_response = json.loads(
                    extract_json_block_content(content))
                if "action" not in parsed_response or "parameters" not in parsed_response:
                    raise ValueError("Invalid LLM response format")
                return parsed_response
            except json.JSONDecodeError:
                raise ValueError(
                    f"Failed to parse LLM response as JSON: {response}")
        except Exception as e:
            logger.error(f"LLM inference failed: {str(e)}")
            raise ValueError(f"Failed to parse instruction: {str(e)}")

    async def _navigate(self, parameters: List[str]) -> Dict[str, Any]:
        url = parameters[0]
        logger.info(f"Navigating to {url}")
        await self.page.goto(url, timeout=60000)
        return {"status": "success", "url": url}

    async def _click(self, parameters: List[str]) -> Dict[str, Any]:
        selector = parameters[0]
        logger.info(f"Clicking element with selector: {selector}")
        await self.page.get_by_role(selector).click(timeout=10000)
        return {"status": "success", "selector": selector}

    async def _fill(self, parameters: List[str]) -> Dict[str, Any]:
        selector, value = parameters
        logger.info(f"Filling {selector} with {value}")
        await self.page.get_by_role(selector).fill(value, timeout=10000)
        return {"status": "success", "selector": selector, "value": value}

    async def _extract_text(self, parameters: List[str]) -> Dict[str, Any]:
        selector = parameters[0]
        logger.info(f"Extracting text from {selector}")
        text = await self.page.get_by_role(selector).inner_text(timeout=10000)
        return {"status": "success", "selector": selector, "text": text}

    async def execute(self, command: str) -> Dict[str, Any]:
        """Executes a single natural language command using LLM parsing."""
        try:
            llm_response = await self._call_llm(command)
            action = llm_response.get("action")
            parameters = llm_response.get("parameters", [])
            if action not in self.action_handlers:
                raise ValueError(f"Unknown action from LLM: {action}")
            handler = self.action_handlers[action]
            return await handler(parameters)
        except Exception as e:
            logger.error(f"Error executing '{command}': {str(e)}")
            return {"status": "error", "instruction": command, "error": str(e)}


async def run_llm_controlled_browser(instructions: List[str]) -> List[Dict[str, Any]]:
    """Runs a sequence of LLM-like instructions using Playwright."""
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        parser = LLMCommandParser(page)

        for instruction in instructions:
            try:
                result = await parser.execute(instruction)
                results.append(result)
            except Exception as e:
                logger.error(f"Error executing '{instruction}': {str(e)}")
                results.append(
                    {"status": "error", "instruction": instruction, "error": str(e)})

        await browser.close()
    return results

if __name__ == "__main__":
    import asyncio
    sample_instructions = [
        "navigate to https://the-internet.herokuapp.com",
        "click link named Login",
        "fill textbox named Username with tomsmith",
        "fill textbox named Password with SuperSecretPassword!",
        "click button named Login",
        "extract text from .flash.success"
    ]
    results = asyncio.run(run_llm_controlled_browser(sample_instructions))
    print(json.dumps(results, indent=2))
