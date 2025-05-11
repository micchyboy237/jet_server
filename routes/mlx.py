import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import requests
from jet.transformers.formatters import format_json
from jet.logger import logger
from jet.llm.mlx.generation import (
    chat,
    generate,
    list_models,
    ChatCompletionRequest,
    TextCompletionRequest,
    UnifiedCompletionResponse
)
import time
from model_cache import MODEL_CACHE, MODEL_CACHE_LOCK

router = APIRouter()


class ModelInfo(BaseModel):
    id: str
    short_name: str
    object: str = "model"
    created: int


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


@router.post("/chat")
async def chat_endpoint(request: ChatCompletionRequest) -> UnifiedCompletionResponse:
    try:
        response = chat(request)
        if request.stream:
            def stream_response():
                for chunk in response:
                    response_dict = {
                        'id': chunk.id,
                        'created': chunk.created,
                        'content': chunk.content,
                        'finish_reason': chunk.finish_reason,
                        'prompt_id': chunk.prompt_id,
                        'task_id': chunk.task_id
                    }
                    if chunk.usage:
                        response_dict['usage'] = chunk.usage.dict()
                    yield f"{format_json(response_dict)}\n"
            return StreamingResponse(stream_response(), media_type="application/json")
        return response
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.text if e.response else str(e)
        logger.error(f"HTTP error from MLX server: {error_detail}")
        raise HTTPException(
            status_code=e.response.status_code if e.response else 500,
            detail=f"MLX LM server error: {error_detail}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal error: {str(e)}")


@router.post("/generate")
async def generate_endpoint(request: TextCompletionRequest) -> UnifiedCompletionResponse:
    try:
        response = generate(request)
        if request.stream:
            def stream_response():
                for chunk in response:
                    response_dict = {
                        'id': chunk.id,
                        'created': chunk.created,
                        'content': chunk.content,
                        'finish_reason': chunk.finish_reason,
                        'prompt_id': chunk.prompt_id,
                        'task_id': chunk.task_id
                    }
                    if chunk.usage:
                        response_dict['usage'] = chunk.usage.dict()
                    yield f"{format_json(response_dict)}\n"
            return StreamingResponse(stream_response(), media_type="application/json")
        return response
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.text if e.response else str(e)
        logger.error(f"HTTP error from MLX server: {error_detail}")
        raise HTTPException(
            status_code=e.response.status_code if e.response else 500,
            detail=f"MLX LM server error: {error_detail}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal error: {str(e)}")


@router.get("/models")
async def models_endpoint() -> ModelListResponse:
    try:
        async with MODEL_CACHE_LOCK:
            if "models" in MODEL_CACHE:
                logger.info("Returning cached model list")
                return ModelListResponse(**MODEL_CACHE["models"])
        response = list_models()
        model_list = ModelListResponse(**response.dict())
        async with MODEL_CACHE_LOCK:
            MODEL_CACHE["models"] = model_list.dict()
        return model_list
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.text if e.response else str(e)
        logger.error(f"HTTP error from MLX server: {error_detail}")
        raise HTTPException(
            status_code=e.response.status_code if e.response else 500,
            detail=f"MLX LM server error: {error_detail}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Models endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal error: {str(e)}")
