import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Iterable, Optional, Dict, Any
import requests
from jet.transformers.formatters import format_json
from jet.logger import logger
from jet.llm.mlx.generation import (
    chat,
    generate,
    get_models,
)
from jet.llm.mlx.mlx_class_types import (
    ChatCompletionRequest,
    TextCompletionRequest,
    UnifiedCompletionResponse,
    Usage,
)
from jet.models.utils import resolve_model_key
import time

router = APIRouter()


class ModelInfo(BaseModel):
    id: str
    short_name: str
    object: str = "model"
    created: int | float
    modified: int | float


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


@router.post("/chat")
async def chat_endpoint(request: ChatCompletionRequest) -> UnifiedCompletionResponse:
    try:
        # Convert Message objects to dictionaries if messages is a list
        messages = request.messages
        if isinstance(messages, list):
            if all(isinstance(msg, list) for msg in messages):
                messages = [[{"role": m.role, "content": m.content}
                             for m in msg_list] for msg_list in messages]
            else:
                messages = [{"role": msg.role, "content": msg.content}
                            for msg in messages]
        response = chat(
            messages=messages,
            model=request.model,
            draft_model=request.draft_model,
            adapter=request.adapters,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            repetition_context_size=request.repetition_context_size,
            xtc_probability=request.xtc_probability,
            xtc_threshold=request.xtc_threshold,
            logit_bias=request.logit_bias,
            logprobs=request.logprobs,
            stop=request.stop,
            role_mapping=request.role_mapping,
            tools=request.tools,
            verbose=request.verbose,
            chat_template_args={"system_prompt": request.system_prompt},
            seed=None,
            prompt_cache=None
        )
        # Convert dictionary response to UnifiedCompletionResponse
        if isinstance(response, dict):
            usage = response.get('usage')
            if usage:
                usage = Usage(
                    prompt_tokens=usage.get('prompt_tokens', 0),
                    prompt_tps=float(usage.get('prompt_tps', '0').split()[0]),
                    completion_tokens=usage.get('completion_tokens', 0),
                    completion_tps=float(
                        usage.get('completion_tps', '0').split()[0]),
                    total_tokens=usage.get('total_tokens', 0),
                    peak_memory=float(usage.get('peak_memory', '0').split()[0])
                )
            response = UnifiedCompletionResponse(
                id=response.get('id', ''),
                created=int(response.get('created', int(time.time()))),
                content=response.get('content'),
                finish_reason=response.get('finish_reason'),
                usage=usage,
                prompt_id=response.get('prompt_id'),
                task_id=response.get('task_id')
            )
        if request.stream:
            if not isinstance(response, Iterable) or isinstance(response, (str, bytes)):
                logger.error(
                    f"Expected an iterable response for streaming, got {type(response)}")
                raise HTTPException(
                    status_code=500, detail="Invalid streaming response: response is not iterable")

            def stream_response():
                try:
                    for chunk in response:
                        try:
                            # Debug log to inspect tuple structure
                            logger.debug(f"Chunk: {chunk}")
                            # Handle tuple or dictionary chunk
                            if isinstance(chunk, tuple):
                                # Flexible tuple parsing, assuming minimal fields
                                chunk_dict = {
                                    'id': chunk[0] if len(chunk) > 0 and isinstance(chunk[0], str) and chunk[0].startswith('chatcmpl-') else '',
                                    'created': int(time.time()),
                                    'content': chunk[1] if len(chunk) > 1 and isinstance(chunk[1], str) else (chunk[0] if len(chunk) > 0 and isinstance(chunk[0], str) and not chunk[0].startswith('chatcmpl-') else None),
                                    'finish_reason': None,  # Default to None if invalid
                                    'usage': chunk[2] if len(chunk) > 2 and isinstance(chunk[2], dict) else None,
                                    'prompt_id': chunk[3] if len(chunk) > 3 else None,
                                    'task_id': chunk[4] if len(chunk) > 4 else None
                                }
                                # Validate finish_reason if provided
                                if len(chunk) > 2 and isinstance(chunk[2], str) and chunk[2] in ('stop', 'length'):
                                    chunk_dict['finish_reason'] = chunk[2]
                                elif len(chunk) > 1 and isinstance(chunk[1], str) and chunk[1] in ('stop', 'length') and not chunk_dict['id']:
                                    chunk_dict['finish_reason'] = chunk[1]
                                    chunk_dict['content'] = chunk[0] if len(
                                        chunk) > 0 and isinstance(chunk[0], str) else None
                                    chunk_dict['usage'] = chunk[2] if len(
                                        chunk) > 2 and isinstance(chunk[2], dict) else None
                                    chunk_dict['prompt_id'] = chunk[3] if len(
                                        chunk) > 3 else None
                                    chunk_dict['task_id'] = chunk[4] if len(
                                        chunk) > 4 else None
                                usage = chunk_dict['usage']
                                if usage:
                                    usage = Usage(
                                        prompt_tokens=usage.get(
                                            'prompt_tokens', 0),
                                        prompt_tps=float(
                                            usage.get('prompt_tps', '0').split()[0]),
                                        completion_tokens=usage.get(
                                            'completion_tokens', 0),
                                        completion_tps=float(
                                            usage.get('completion_tps', '0').split()[0]),
                                        total_tokens=usage.get(
                                            'total_tokens', 0),
                                        peak_memory=float(
                                            usage.get('peak_memory', '0').split()[0])
                                    )
                                chunk = UnifiedCompletionResponse(
                                    id=chunk_dict['id'],
                                    created=chunk_dict['created'],
                                    content=chunk_dict['content'],
                                    finish_reason=chunk_dict['finish_reason'],
                                    usage=usage,
                                    prompt_id=chunk_dict['prompt_id'],
                                    task_id=chunk_dict['task_id']
                                )
                            elif isinstance(chunk, dict):
                                usage = chunk.get('usage')
                                if usage:
                                    usage = Usage(
                                        prompt_tokens=usage.get(
                                            'prompt_tokens', 0),
                                        prompt_tps=float(
                                            usage.get('prompt_tps', '0').split()[0]),
                                        completion_tokens=usage.get(
                                            'completion_tokens', 0),
                                        completion_tps=float(
                                            usage.get('completion_tps', '0').split()[0]),
                                        total_tokens=usage.get(
                                            'total_tokens', 0),
                                        peak_memory=float(
                                            usage.get('peak_memory', '0').split()[0])
                                    )
                                chunk_dict = chunk
                                chunk = UnifiedCompletionResponse(
                                    id=chunk.get('id', ''),
                                    created=int(
                                        chunk.get('created', int(time.time()))),
                                    content=chunk.get('content'),
                                    finish_reason=chunk.get('finish_reason') if chunk.get(
                                        'finish_reason') in ('stop', 'length') else None,
                                    usage=usage,
                                    prompt_id=chunk.get('prompt_id'),
                                    task_id=chunk.get('task_id')
                                )
                            response_dict = {
                                'id': chunk_dict['id'],
                                'created': chunk_dict['created'],
                                'content': chunk_dict['content'],
                                'finish_reason': chunk_dict['finish_reason'],
                                'prompt_id': chunk_dict['prompt_id'],
                                'task_id': chunk_dict['task_id']
                            }
                            if chunk.usage:
                                response_dict['usage'] = chunk.usage.dict()
                            yield f"{format_json(response_dict)}\n"
                        except Exception as e:
                            logger.error(
                                f"Error processing chunk: {str(e)}", exc_info=True)
                            yield f"{format_json({'error': f'Chunk processing error: {str(e)}'})}\n"
                except Exception as e:
                    logger.error(
                        f"Error iterating response: {str(e)}", exc_info=True)
                    yield f"{format_json({'error': f'Stream iteration error: {str(e)}'})}\n"
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
        response = generate(
            prompt=request.prompt,
            model=request.model,
            draft_model=request.draft_model,
            adapter=request.adapters,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            repetition_context_size=request.repetition_context_size,
            xtc_probability=request.xtc_probability,
            xtc_threshold=request.xtc_threshold,
            logit_bias=request.logit_bias,
            logprobs=request.logprobs,
            stop=request.stop,
            verbose=request.verbose,
            seed=None,
            prompt_cache=None
        )
        # Convert dictionary response to UnifiedCompletionResponse
        if isinstance(response, dict):
            usage = response.get('usage')
            if usage:
                usage = Usage(
                    prompt_tokens=usage.get('prompt_tokens', 0),
                    prompt_tps=float(usage.get('prompt_tps', '0').split()[0]),
                    completion_tokens=usage.get('completion_tokens', 0),
                    completion_tps=float(
                        usage.get('completion_tps', '0').split()[0]),
                    total_tokens=usage.get('total_tokens', 0),
                    peak_memory=float(usage.get('peak_memory', '0').split()[0])
                )
            response = UnifiedCompletionResponse(
                id=response.get('id', ''),
                created=int(response.get('created', int(time.time()))),
                content=response.get('content'),
                finish_reason=response.get('finish_reason'),
                usage=usage,
                prompt_id=response.get('prompt_id'),
                task_id=response.get('task_id')
            )
        if request.stream:
            if not isinstance(response, Iterable) or isinstance(response, (str, bytes)):
                logger.error(
                    f"Expected an iterable response for streaming, got {type(response)}")
                raise HTTPException(
                    status_code=500, detail="Invalid streaming response: response is not iterable")

            def stream_response():
                try:
                    for chunk in response:
                        try:
                            # Debug log to inspect tuple structure
                            logger.debug(f"Chunk: {chunk}")
                            # Handle tuple or dictionary chunk
                            if isinstance(chunk, tuple):
                                # Flexible tuple parsing, assuming minimal fields
                                chunk_dict = {
                                    'id': chunk[0] if len(chunk) > 0 and isinstance(chunk[0], str) and chunk[0].startswith('chatcmpl-') else '',
                                    'created': int(time.time()),
                                    'content': chunk[1] if len(chunk) > 1 and isinstance(chunk[1], str) else (chunk[0] if len(chunk) > 0 and isinstance(chunk[0], str) and not chunk[0].startswith('chatcmpl-') else None),
                                    'finish_reason': None,  # Default to None if invalid
                                    'usage': chunk[2] if len(chunk) > 2 and isinstance(chunk[2], dict) else None,
                                    'prompt_id': chunk[3] if len(chunk) > 3 else None,
                                    'task_id': chunk[4] if len(chunk) > 4 else None
                                }
                                # Validate finish_reason if provided
                                if len(chunk) > 2 and isinstance(chunk[2], str) and chunk[2] in ('stop', 'length'):
                                    chunk_dict['finish_reason'] = chunk[2]
                                elif len(chunk) > 1 and isinstance(chunk[1], str) and chunk[1] in ('stop', 'length') and not chunk_dict['id']:
                                    chunk_dict['finish_reason'] = chunk[1]
                                    chunk_dict['content'] = chunk[0] if len(
                                        chunk) > 0 and isinstance(chunk[0], str) else None
                                    chunk_dict['usage'] = chunk[2] if len(
                                        chunk) > 2 and isinstance(chunk[2], dict) else None
                                    chunk_dict['prompt_id'] = chunk[3] if len(
                                        chunk) > 3 else None
                                    chunk_dict['task_id'] = chunk[4] if len(
                                        chunk) > 4 else None
                                usage = chunk_dict['usage']
                                if usage:
                                    usage = Usage(
                                        prompt_tokens=usage.get(
                                            'prompt_tokens', 0),
                                        prompt_tps=float(
                                            usage.get('prompt_tps', '0').split()[0]),
                                        completion_tokens=usage.get(
                                            'completion_tokens', 0),
                                        completion_tps=float(
                                            usage.get('completion_tps', '0').split()[0]),
                                        total_tokens=usage.get(
                                            'total_tokens', 0),
                                        peak_memory=float(
                                            usage.get('peak_memory', '0').split()[0])
                                    )
                                chunk = UnifiedCompletionResponse(
                                    id=chunk_dict['id'],
                                    created=chunk_dict['created'],
                                    content=chunk_dict['content'],
                                    finish_reason=chunk_dict['finish_reason'],
                                    usage=usage,
                                    prompt_id=chunk_dict['prompt_id'],
                                    task_id=chunk_dict['task_id']
                                )
                            elif isinstance(chunk, dict):
                                usage = chunk.get('usage')
                                if usage:
                                    usage = Usage(
                                        prompt_tokens=usage.get(
                                            'prompt_tokens', 0),
                                        prompt_tps=float(
                                            usage.get('prompt_tps', '0').split()[0]),
                                        completion_tokens=usage.get(
                                            'completion_tokens', 0),
                                        completion_tps=float(
                                            usage.get('completion_tps', '0').split()[0]),
                                        total_tokens=usage.get(
                                            'total_tokens', 0),
                                        peak_memory=float(
                                            usage.get('peak_memory', '0').split()[0])
                                    )
                                chunk_dict = chunk
                                chunk = UnifiedCompletionResponse(
                                    id=chunk.get('id', ''),
                                    created=int(
                                        chunk.get('created', int(time.time()))),
                                    content=chunk.get('content'),
                                    finish_reason=chunk.get('finish_reason') if chunk.get(
                                        'finish_reason') in ('stop', 'length') else None,
                                    usage=usage,
                                    prompt_id=chunk.get('prompt_id'),
                                    task_id=chunk.get('task_id')
                                )
                            response_dict = {
                                'id': chunk_dict['id'],
                                'created': chunk_dict['created'],
                                'content': chunk_dict['content'],
                                'finish_reason': chunk_dict['finish_reason'],
                                'prompt_id': chunk_dict['prompt_id'],
                                'task_id': chunk_dict['task_id']
                            }
                            if chunk.usage:
                                response_dict['usage'] = chunk.usage.dict()
                            yield f"{format_json(response_dict)}\n"
                        except Exception as e:
                            logger.error(
                                f"Error processing chunk: {str(e)}", exc_info=True)
                            yield f"{format_json({'error': f'Chunk processing error: {str(e)}'})}\n"
                except Exception as e:
                    logger.error(
                        f"Error iterating response: {str(e)}", exc_info=True)
                    yield f"{format_json({'error': f'Stream iteration error: {str(e)}'})}\n"
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
        models = get_models()
        return ModelListResponse(
            data=[ModelInfo(short_name=resolve_model_key(model_info['id']), **model_info)
                  for model_info in models['data']]
        )
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
