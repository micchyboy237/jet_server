from typing import Generator
from fastapi import APIRouter, HTTPException
from jet.transformers.formatters import format_json
from pydantic import BaseModel
from jet.llm.ollama.base import Ollama
from jet.llm.main.prompts_generator import PromptsGenerator
from jet.logger import logger
from starlette.responses import StreamingResponse
from tqdm import tqdm

# FastAPI router
router = APIRouter()

# Request model


class PromptsRequest(BaseModel):
    prompts: list[str]

# Response model


# class PromptResponse(BaseModel):
#     prompt: str
#     results: list[str]

def event_stream(prompts: list[str]) -> Generator[str, None, None]:
    """Generator function to yield events for streaming."""
    try:
        processor = PromptsGenerator(llm=Ollama(model="llama3.1"))
        response_stream = processor.process(prompts)

        generation_tqdm = tqdm(response_stream)

        for (text, result) in generation_tqdm:
            yield f"data: {result}\n\n"
            logger.success(result)

            # Update the progress bar after processing each node
            generation_tqdm.update(1)

    except Exception as e:
        logger.error(f"Error processing prompts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-prompts", response_model=list[str])
async def generate_prompts(request: PromptsRequest):
    """
    Process a list of prompts and return generated responses.
    """

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
    }
    return StreamingResponse(event_stream(request.prompts), headers=headers)
