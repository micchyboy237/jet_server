from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from jet.logger import logger
from jet.transformers.object import make_serializable
from pydantic import BaseModel
from mlx_lm import load, generate, stream_generate
from models_config import AVAILABLE_MODELS
import json
import gc
import mlx.core as mx
import asyncio

router = APIRouter()

# Global cache to store the currently loaded model and tokenizer
MODEL_CACHE = {
    "model": None,
    "tokenizer": None,
    "model_name": None
}

# Lock to ensure thread-safe model loading/unloading
MODEL_CACHE_LOCK = asyncio.Lock()


class TextGenerationRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 300
    with_info: bool = False


class TextGenerationResponse(BaseModel):
    generated_text: str


def unload_current_model():
    """Unload the currently cached model to free RAM."""
    if MODEL_CACHE["model"] is not None:
        logger.info(f"Unloading model: {MODEL_CACHE['model_name']}")
        # Delete references to model and tokenizer
        MODEL_CACHE["model"] = None
        MODEL_CACHE["tokenizer"] = None
        MODEL_CACHE["model_name"] = None
        # Force garbage collection and clear MLX memory
        gc.collect()
        mx.metal.clear_cache()  # Clear MLX metal cache
        logger.info("Model unloaded and memory cleared.")


async def load_model(model_name: str):
    """Load a model and update the cache, unloading the current model if necessary."""
    async with MODEL_CACHE_LOCK:
        if model_name != MODEL_CACHE["model_name"]:
            # Unload the current model if a different model is requested
            unload_current_model()
            model_path = AVAILABLE_MODELS[model_name]
            logger.info(f"Loading model: {model_path}")
            model, tokenizer = load(model_path)
            # Update the cache
            MODEL_CACHE["model"] = model
            MODEL_CACHE["tokenizer"] = tokenizer
            MODEL_CACHE["model_name"] = model_name
            logger.info(f"Model {model_name} loaded and cached.")
        return MODEL_CACHE["model"], MODEL_CACHE["tokenizer"]


async def stream_tokens(model, tokenizer, prompt, max_tokens, with_info: bool = False):
    """Generator function to stream tokens from stream_generate."""
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    ):
        logger.success(response.text, flush=True)
        # Ensure newlines are preserved in the response text
        text = response.text.replace("\r\n", "\n")  # Normalize newlines
        if with_info:
            yield json.dumps(make_serializable(response)) + "\n"
        else:
            yield f"data: {text}\n\n"


@router.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    try:
        if request.model not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Available models: {list(AVAILABLE_MODELS.keys())}"
            )
        model, tokenizer = await load_model(request.model)

        logger.info(f"Generating text with model: {request.model}")
        logger.log("\nPrompt:", request.prompt, colors=["GRAY", "DEBUG"])

        response = generate(
            model,
            tokenizer,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            verbose=True
        )
        return TextGenerationResponse(generated_text=response)
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_text(request: TextGenerationRequest):
    try:
        if request.model not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Available models: {list(AVAILABLE_MODELS.keys())}"
            )
        model, tokenizer = await load_model(request.model)

        logger.info(f"Streaming text with model: {request.model}")
        logger.log("\nPrompt:", request.prompt, colors=["GRAY", "DEBUG"])

        media_type = "application/x-ndjson" if request.with_info else "text/event-stream"
        return StreamingResponse(
            stream_tokens(model, tokenizer, request.prompt,
                          request.max_tokens, request.with_info),
            media_type=media_type
        )
    except Exception as e:
        logger.error(f"Error streaming text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=TextGenerationResponse)
async def chat_text(request: TextGenerationRequest):
    try:
        if request.model not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Available models: {list(AVAILABLE_MODELS.keys())}"
            )
        model, tokenizer = await load_model(request.model)
        prompt = request.prompt

        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )

        logger.info(f"Chat generation with model: {request.model}")
        logger.log("\nChat prompt:", request.prompt, colors=["GRAY", "DEBUG"])

        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=request.max_tokens,
            verbose=True
        )
        return TextGenerationResponse(generated_text=response)
    except Exception as e:
        logger.error(f"Error in chat generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
