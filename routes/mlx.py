from jet.transformers.formatters import format_json
import mlx.core as mx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
from fastapi.responses import StreamingResponse
from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler
from model_cache import MODEL_CACHE, MODEL_CACHE_LOCK, load_model
from jet.llm.mlx.utils import get_model_max_tokens
from jet.logger import logger
import time

router = APIRouter()


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
        "num_predict": int,
        "num_keep": int
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
                if key == "num_predict":
                    processed["max_tokens"] = value
                elif key == "num_keep":
                    sampler_params["min_tokens_to_keep"] = value
                else:
                    processed[key] = value
            elif key in sampler_options and isinstance(value, sampler_options[key]):
                sampler_params[key] = value
    if processed.get("seed") is not None:
        mx.random.seed(processed["seed"])
    temp = sampler_params.get("temperature", 0.8)
    top_p = sampler_params.get("top_p", 0.9)
    min_p = sampler_params.get("min_p", 0.0)
    min_tokens_to_keep = sampler_params.get(
        "min_tokens_to_keep", 1)  # Default to 1 if not set
    sampler = make_sampler(temp, top_p, min_p, min_tokens_to_keep)
    return processed, sampler


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        model, tokenizer = await load_model(request.model)
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

            model_max_tokens = get_model_max_tokens(request.model)

            logger.newline()
            logger.log("Options:", format_json(
                options), colors=["GRAY", "DEBUG"])
            logger.log("Model:", request.model, colors=["GRAY", "DEBUG"])

            logger.newline()
            logger.log("Prompt Tokens:", len(prompt_tokens),
                       colors=["GRAY", "ORANGE"])
            logger.log("Model Max Tokens:", model_max_tokens,
                       colors=["GRAY", "INFO"])
            remaining_tokens = model_max_tokens - len(prompt_tokens)
            remaining_percentage = round(
                (remaining_tokens / model_max_tokens) * 100, 2)
            logger.log("Remaining Tokens:", f"{remaining_tokens} ({remaining_percentage}%)", colors=[
                       "GRAY", "INFO"])

            async def stream_response():
                async with MODEL_CACHE_LOCK:
                    MODEL_CACHE["last_used"] = time.time()
                for response in stream_generate(
                    model,
                    tokenizer,
                    prompt_tokens,
                    sampler=sampler,
                    **options
                ):
                    logger.success(response.text, flush=True)
                    yield f"data: {response.text}\n\n"

                    if response.finish_reason:
                        logger.newline()
                        total_tokens = response.prompt_tokens + response.generation_tokens
                        total_tps = response.prompt_tps + response.generation_tps
                        logger.log(
                            "Prompt:",
                            f"{response.prompt_tokens} tokens,",
                            f"{response.prompt_tps:.3f} tokens-per-sec",
                            colors=["GRAY", "ORANGE", "GRAY"]
                        )
                        logger.log(
                            "Generation:",
                            f"{response.generation_tokens} tokens,",
                            f"{response.generation_tps:.3f} tokens-per-sec",
                            colors=["GRAY", "ORANGE", "GRAY"]
                        )
                        logger.log(
                            f"Peak memory:",
                            f"{response.peak_memory:.3f}",
                            "GB",
                            colors=["GRAY", "ORANGE", "GRAY"]
                        )
                        logger.log(
                            "Total:",
                            f"{total_tokens} tokens,",
                            f"{total_tps:.3f} tokens-per-sec",
                            colors=["GRAY", "SUCCESS", "GRAY"]
                        )
                        logger.newline()
                        logger.log("Model Max Tokens:",
                                   model_max_tokens, colors=["GRAY", "INFO"])
                        remaining_tokens = model_max_tokens - total_tokens
                        remaining_percentage = round(
                            (remaining_tokens / model_max_tokens) * 100, 2)
                        logger.log("Remaining Tokens:", f"{remaining_tokens} ({remaining_percentage}%)", colors=[
                                   "GRAY", "INFO"])
                        logger.newline()

                    async with MODEL_CACHE_LOCK:
                        MODEL_CACHE["last_used"] = time.time()
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
                verbose=True,
                **options
            )
            async with MODEL_CACHE_LOCK:
                MODEL_CACHE["last_used"] = time.time()
            logger.success(response)
            final_response = GenerationResponse(
                text=response,
                token=0,
                logprobs={},
                from_draft=False,
                prompt_tokens=len(prompt_tokens),
                prompt_tps=0.0,
                generation_tokens=0,
                generation_tps=0.0,
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason="stop"
            )
            return final_response
    except Exception as e:
        logger.error(f"Error in chat generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    try:
        model, tokenizer = await load_model(request.model)
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
            logger.info(f"Streaming text with model: {request.model}")
            logger.log("\nPrompt:", prompt, colors=["GRAY", "DEBUG"])

            async def stream_response():
                async with MODEL_CACHE_LOCK:
                    MODEL_CACHE["last_used"] = time.time()
                for response in stream_generate(
                    model,
                    tokenizer,
                    prompt_tokens,
                    sampler=sampler,
                    **options
                ):
                    logger.success(response.text, flush=True)
                    yield f"data: {response.text}\n\n"
                    async with MODEL_CACHE_LOCK:
                        MODEL_CACHE["last_used"] = time.time()
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
                verbose=True,
                **options
            )
            async with MODEL_CACHE_LOCK:
                MODEL_CACHE["last_used"] = time.time()
            final_response = GenerationResponse(
                text=response,
                token=0,
                logprobs={},
                from_draft=False,
                prompt_tokens=len(prompt_tokens),
                prompt_tps=0.0,
                generation_tokens=0,
                generation_tps=0.0,
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason="stop"
            )
            return final_response
    except Exception as e:
        logger.error(f"Error in text generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
