from typing import Optional
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
import time

router = APIRouter()

MODEL_CACHE = {
    "model": None,
    "tokenizer": None,
    "model_name": None,
    "last_used": None  # Track last usage time
}
MODEL_CACHE_LOCK = asyncio.Lock()


class TextGenerationRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 300
    with_info: bool = False
    stop: list[str] = []


class TextGenerationResponse(BaseModel):
    generated_text: str


def unload_current_model():
    """Unload the currently cached model to free RAM."""
    if MODEL_CACHE["model"] is not None:
        logger.info(f"Unloading model: {MODEL_CACHE['model_name']}")
        MODEL_CACHE["model"] = None
        MODEL_CACHE["tokenizer"] = None
        MODEL_CACHE["model_name"] = None
        MODEL_CACHE["last_used"] = None
        mx.clear_cache()
        gc.collect()
        logger.info("Model unloaded and memory cleared.")


async def load_model(model_name: str):
    """Load a model and update the cache, unloading the current model if necessary."""
    async with MODEL_CACHE_LOCK:
        if model_name != MODEL_CACHE["model_name"]:
            unload_current_model()
            model_path = AVAILABLE_MODELS[model_name]
            logger.info(f"Loading model: {model_path}")
            model, tokenizer = load(model_path)
            MODEL_CACHE["model"] = model
            MODEL_CACHE["tokenizer"] = tokenizer
            MODEL_CACHE["model_name"] = model_name
            MODEL_CACHE["last_used"] = time.time()  # Update last used time
            logger.info(f"Model {model_name} loaded and cached.")
        else:
            # Update last used time if model is already loaded
            MODEL_CACHE["last_used"] = time.time()
        return MODEL_CACHE["model"], MODEL_CACHE["tokenizer"]


async def cleanup_idle_models():
    """Background task to unload models idle for more than 1 minute."""
    while True:
        async with MODEL_CACHE_LOCK:
            if MODEL_CACHE["model"] is not None and MODEL_CACHE["last_used"] is not None:
                idle_time = time.time() - MODEL_CACHE["last_used"]
                if idle_time > 60:  # 1 minute in seconds
                    logger.info(
                        f"Model {MODEL_CACHE['model_name']} idle for {idle_time:.2f} seconds, unloading.")
                    unload_current_model()
        await asyncio.sleep(10)  # Check every 10 seconds


async def stream_tokens(model, tokenizer, prompt, max_tokens, with_info: bool = False, stop: list[str] = []):
    """Generator function to stream tokens from stream_generate with stop sequence handling."""
    async with MODEL_CACHE_LOCK:
        # Update last used time at start
        MODEL_CACHE["last_used"] = time.time()

    accumulated_text = ""
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens
    ):
        logger.success(response.text, flush=True)
        text = response.text.replace("\r\n", "\n")
        accumulated_text += text

        # Check for stop sequences
        stop_detected = False
        for stop_seq in stop:
            if stop_seq in accumulated_text:
                # Truncate accumulated text up to the stop sequence
                stop_index = accumulated_text.index(stop_seq)
                text_to_yield = accumulated_text[:stop_index]
                accumulated_text = text_to_yield  # Update accumulated text
                stop_detected = True
                break

        # Yield the current text
        if with_info:
            yield json.dumps(make_serializable(response)) + "\n"
        else:
            yield f"data: {text}\n\n"

        # Break if a stop sequence was detected
        if stop_detected:
            break

    async with MODEL_CACHE_LOCK:
        MODEL_CACHE["last_used"] = time.time()  # Update last used time at end


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
        for stop in request.stop:
            tokenizer.add_eos_token(stop)
        response = generate(
            model,
            tokenizer,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            stop=request.stop,
            verbose=True
        )
        async with MODEL_CACHE_LOCK:
            MODEL_CACHE["last_used"] = time.time()  # Update last used time
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
            stream_tokens(
                model,
                tokenizer,
                request.prompt,
                request.max_tokens,
                request.with_info,
                stop=request.stop
            ),
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
        for stop in request.stop:
            tokenizer.add_eos_token(stop)
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=request.max_tokens,
            stop=request.stop,
            verbose=True
        )
        async with MODEL_CACHE_LOCK:
            MODEL_CACHE["last_used"] = time.time()  # Update last used time
        return TextGenerationResponse(generated_text=response)
    except Exception as e:
        logger.error(f"Error in chat generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
