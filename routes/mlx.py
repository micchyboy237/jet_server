import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import requests
from jet.transformers.formatters import format_json
from jet.logger import logger
from jet.llm.mlx.generation import (
    chat_completions,
    text_completions,
    list_models,
    ChatCompletionRequest,
    TextCompletionRequest,
    UnifiedCompletionResponse
)
import time
from model_cache import MODEL_CACHE, MODEL_CACHE_LOCK

router = APIRouter()


@router.post("/chat")
async def chat_endpoint(request: ChatCompletionRequest):
    try:
        response = chat_completions(request)
        if request.stream:
            def stream_response():
                for chunk in response:  # Iterate over generator for streaming chunks
                    yield format_json({
                        "id": chunk.id,
                        "created": chunk.created,
                        "content": chunk.content,
                        "finish_reason": chunk.finish_reason
                    }) + "\n"
            return StreamingResponse(stream_response(), media_type="application/x-ndjson")
        return response
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error from MLX server: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"MLX LM server error: {str(e)}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/generate")
async def generate_endpoint(request: TextCompletionRequest):
    try:
        response = text_completions(request)
        if request.stream:
            def stream_response():
                for chunk in response:  # Iterate over generator for streaming chunks
                    yield format_json({
                        "id": chunk.id,
                        "created": chunk.created,
                        "content": chunk.content,
                        "finish_reason": chunk.finish_reason
                    }) + "\n"
            return StreamingResponse(stream_response(), media_type="application/x-ndjson")
        return response
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error from MLX server: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"MLX LM server error: {str(e)}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/models")
async def models_endpoint():
    try:
        async with MODEL_CACHE_LOCK:
            if "models" in MODEL_CACHE:
                logger.info("Returning cached model list")
                return MODEL_CACHE["models"]
        response = list_models()
        async with MODEL_CACHE_LOCK:
            MODEL_CACHE["models"] = response
        return response
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error from MLX server: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"MLX LM server error: {str(e)}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Models endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Internal error: {str(e)}")
