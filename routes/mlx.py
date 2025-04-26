import mlx.core as mx
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
from fastapi.responses import StreamingResponse
from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler
from models_config import AVAILABLE_MODELS
from jet.logger import logger

router = APIRouter()

# Pydantic models for request validation


class Message(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None
    tool_calls: Optional[List[Dict]] = None


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    tools: Optional[List[Dict]] = None
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = True
    keep_alive: Optional[str] = "5m"


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    suffix: Optional[str] = None
    images: Optional[List[str]] = None
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    system: Optional[str] = None
    template: Optional[str] = None
    stream: Optional[bool] = True
    raw: Optional[bool] = False
    keep_alive: Optional[str] = "5m"
    context: Optional[List[int]] = None


class GenerationResponse(BaseModel):
    text: str
    token: int
    logprobs: Dict
    from_draft: bool
    prompt_tokens: int
    prompt_tps: float
    generation_tokens: int
    generation_tps: float
    peak_memory: float
    finish_reason: Optional[str] = None

# Helper function to validate and convert options


def process_options(options: Optional[Dict[str, Any]]) -> tuple[Dict[str, Any], Any]:
    valid_options = {
        "mirostat": int,
        "mirostat_eta": float,
        "mirostat_tau": float,
        "num_ctx": int,
        "repeat_last_n": int,
        "repeat_penalty": float,
        "seed": int,
        "stop": str,
        "num_predict": int,  # Will be mapped to max_tokens
        "max_kv_size": int
    }

    sampler_options = {
        "temperature": float,
        "top_p": float,
        "min_p": float,
        "min_tokens_to_keep": int
    }

    processed = {}
    sampler_params = {}

    if options:
        for key, value in options.items():
            if key in valid_options and isinstance(value, valid_options[key]):
                # Map num_predict to max_tokens
                if key == "num_predict":
                    processed["max_tokens"] = value
                else:
                    processed[key] = value
            elif key in sampler_options and isinstance(value, sampler_options[key]):
                sampler_params[key] = value

    # Create sampler with default values if not provided
    temp = sampler_params.get("temperature", 0.8)
    top_p = sampler_params.get("top_p", 0.9)
    min_p = sampler_params.get("min_p", 0.0)
    min_tokens_to_keep = sampler_params.get("min_tokens_to_keep", 1)

    sampler = make_sampler(temp, top_p, min_p, min_tokens_to_keep)

    return processed, sampler

# Load model and tokenizer


def load_model_and_tokenizer(model_name: str):
    model_path = AVAILABLE_MODELS[model_name]
    model, tokenizer = load(model_path)
    return model, tokenizer


@router.post("/chat")
async def chat(request: ChatRequest):
    model, tokenizer = load_model_and_tokenizer(request.model)

    # Apply chat template
    messages = [msg.dict(exclude_none=True) for msg in request.messages]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    options, sampler = process_options(request.options)

    if request.stream:
        logger.info(f"Streaming text with model: {request.model}")
        logger.log("\nPrompt:", prompt, colors=["GRAY", "DEBUG"])

        async def stream_response():
            for response in stream_generate(
                model,
                tokenizer,
                prompt_tokens,
                sampler=sampler,
                **options
            ):
                logger.success(response.text, flush=True)
                yield f"data: {response.text}\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream"
        )
    else:
        logger.info(f"Generating text with model: {request.model}")
        logger.log("\nPrompt:", prompt, colors=["GRAY", "DEBUG"])
        response = generate(
            model,
            tokenizer,
            prompt_tokens,
            sampler=sampler,
            **options
        )
        logger.success(response)
        # Create a final response object
        final_response = GenerationResponse(
            text=response,
            token=0,  # Last token not tracked in non-streaming
            logprobs={},
            from_draft=False,
            prompt_tokens=len(prompt_tokens),
            prompt_tps=0.0,  # Not measured in non-streaming
            generation_tokens=0,  # Not measured in non-streaming
            generation_tps=0.0,  # Not measured in non-streaming
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="stop"
        )
        return final_response


@router.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    model, tokenizer = load_model_and_tokenizer(request.model)

    # Prepare prompt
    prompt = request.prompt
    if request.system:
        messages = [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.prompt}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    if not request.raw and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    options, sampler = process_options(request.options)

    if request.stream:
        async def stream_response():
            for response in stream_generate(
                model,
                tokenizer,
                prompt_tokens,
                sampler=sampler,
                **options
            ):
                logger.success(response.text, flush=True)
                yield f"data: {response.text}\n\n"

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream"
        )
    else:
        response = generate(
            model,
            tokenizer,
            prompt_tokens,
            sampler=sampler,
            **options
        )
        # Create a final response object
        final_response = GenerationResponse(
            text=response,
            token=0,  # Last token not tracked in non-streaming
            logprobs={},
            from_draft=False,
            prompt_tokens=len(prompt_tokens),
            prompt_tps=0.0,  # Not measured in non-streaming
            generation_tokens=0,  # Not measured in non-streaming
            generation_tps=0.0,  # Not measured in non-streaming
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="stop"
        )
        return final_response
