from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from mlx_lm import load, generate, stream_generate
import logging
from models_config import AVAILABLE_MODELS
import json

router = APIRouter()
logger = logging.getLogger(__name__)


class TextGenerationRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 300


class TextGenerationResponse(BaseModel):
    generated_text: str


async def stream_tokens(model, tokenizer, prompt, max_tokens):
    """Generator function to stream tokens from stream_generate."""
    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
    ):
        yield json.dumps({"token": response.text}) + "\n"


@router.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    try:
        # Validate model
        if request.model not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Available models: {list(AVAILABLE_MODELS.keys())}"
            )

        # Load model and tokenizer
        model_path = AVAILABLE_MODELS[request.model]
        logger.info(f"Loading model: {model_path}")
        model, tokenizer = load(model_path)

        # Prepare prompt
        prompt = request.prompt
        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )

        # Generate response
        logger.info(f"Generating text with model: {request.model}")
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=request.max_tokens,
        )

        return TextGenerationResponse(generated_text=response)

    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_text(request: TextGenerationRequest):
    try:
        # Validate model
        if request.model not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model. Available models: {list(AVAILABLE_MODELS.keys())}"
            )

        # Load model and tokenizer
        model_path = AVAILABLE_MODELS[request.model]
        logger.info(f"Loading model: {model_path}")
        model, tokenizer = load(model_path)

        # Prepare prompt
        prompt = request.prompt
        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )

        # Stream response
        logger.info(f"Streaming text with model: {request.model}")
        return StreamingResponse(
            stream_tokens(model, tokenizer, prompt,
                          request.max_tokens),
            media_type="application/x-ndjson"
        )

    except Exception as e:
        logger.error(f"Error streaming text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
