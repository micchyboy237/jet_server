import json
import time
import requests
from typing import List, Dict, Optional, Union, Literal, Generator
from pydantic import BaseModel, Field
from fastapi import HTTPException
from jet.logger import logger

BASE_URL = "http://localhost:8003/v1"

# Request Models


class Message(BaseModel):
    role: str = Field(...,
                      description="Role of the message sender (e.g., 'user', 'assistant')")
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    messages: List[Message] = Field(
        ..., description="Array of message objects representing conversation history")
    temperature: float = Field(1.0, description="Sampling temperature", ge=0.0)
    top_p: float = Field(
        1.0, description="Nucleus sampling parameter", ge=0.0, le=1.0)
    max_tokens: int = Field(
        100, description="Maximum number of tokens to generate", ge=1)
    stream: bool = Field(False, description="Whether to stream the response")
    stop: Optional[Union[str, List[str]]] = Field(
        None, description="Sequences where generation should stop")
    role_mapping: Optional[Dict[str, str]] = Field(
        None, description="Custom role prefixes for prompt generation")
    repetition_penalty: float = Field(
        1.0, description="Penalty for repeated tokens", ge=1.0)
    repetition_context_size: int = Field(
        20, description="Context window size for repetition penalty", ge=1)
    logit_bias: Optional[Dict[int, float]] = Field(
        None, description="Token ID to bias value mapping")
    logprobs: Optional[int] = Field(
        None, description="Number of top tokens and log probabilities to return", ge=1, le=10)
    model: Optional[str] = Field(
        None, description="Path to local model or Hugging Face repo ID")
    adapters: Optional[str] = Field(
        None, description="Path to low-rank adapters")
    draft_model: Optional[str] = Field(
        None, description="Smaller model for speculative decoding")
    num_draft_tokens: int = Field(
        3, description="Number of draft tokens for draft model", ge=1)


class TextCompletionRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for text completion")
    temperature: float = Field(1.0, description="Sampling temperature", ge=0.0)
    top_p: float = Field(
        1.0, description="Nucleus sampling parameter", ge=0.0, le=1.0)
    max_tokens: int = Field(
        100, description="Maximum number of tokens to generate", ge=1)
    stream: bool = Field(False, description="Whether to stream the response")
    stop: Optional[Union[str, List[str]]] = Field(
        None, description="Sequences where generation should stop")
    repetition_penalty: float = Field(
        1.0, description="Penalty for repeated tokens", ge=1.0)
    repetition_context_size: int = Field(
        20, description="Context window size for repetition penalty", ge=1)
    logit_bias: Optional[Dict[int, float]] = Field(
        None, description="Token ID to bias value mapping")
    logprobs: Optional[int] = Field(
        None, description="Number of top tokens and log probabilities to return", ge=1, le=10)
    model: Optional[str] = Field(
        None, description="Path to local model or Hugging Face repo ID")
    adapters: Optional[str] = Field(
        None, description="Path to low-rank adapters")
    draft_model: Optional[str] = Field(
        None, description="Smaller model for speculative decoding")
    num_draft_tokens: int = Field(
        3, description="Number of draft tokens for draft model", ge=1)

# Response Models


class Usage(BaseModel):
    prompt_tokens: int = Field(...,
                               description="Number of prompt tokens processed")
    completion_templates: int = Field(...,
                                      description="Number of tokens generated")
    total_tokens: int = Field(..., description="Total number of tokens")


class LogProbs(BaseModel):
    token_logprobs: List[float] = Field(
        default_factory=list, description="Log probabilities for generated tokens")
    tokens: Optional[List[int]] = Field(
        default=None, description="Generated token IDs")
    top_logprobs: List[Dict[int, float]] = Field(
        default_factory=list, description="Top tokens and their log probabilities")


class Choice(BaseModel):
    index: int = Field(..., description="Index of the choice in the list")
    message: Optional[Message] = Field(
        None, description="Text response from the model")
    text: Optional[str] = Field(
        None, description="Generated text for text completion")
    logprobs: Optional[LogProbs] = Field(
        None, description="Log probabilities for generated tokens")
    finish_reason: Optional[Literal["stop", "length"]] = Field(
        None, description="Reason the completion ended")


class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the chat")
    system_fingerprint: str = Field(...,
                                    description="Unique identifier for the system")
    object: Literal["chat.completion", "chat.completion.chunk", "text.completion",
                    "text.completion.chunk", "text_completion"] = Field(..., description="Type of response")
    model: str = Field(..., description="Model repo or path")
    created: int = Field(...,
                         description="Timestamp for when the request was processed")
    choices: List[Choice] = Field(..., description="List of output choices")
    usage: Optional[Usage] = Field(None, description="Token usage information")


class ModelInfo(BaseModel):
    id: str = Field(..., description="Hugging Face repo ID")
    created: int = Field(..., description="Timestamp for model creation")


class ModelsResponse(BaseModel):
    data: List[ModelInfo] = Field(..., description="List of available models")

# Helper Function


def _handle_response(response: requests.Response, is_stream: bool, object_type: str) -> Union[ChatCompletionResponse, Generator[ChatCompletionResponse, None, None]]:
    """
    Handle streaming and non-streaming responses from the MLX LM server.

    Args:
        response: The HTTP response from the server.
        is_stream: Whether the response is expected to be streaming (text/event-stream).
        object_type: The type of response object ("chat.completion", "text.completion", or their chunk variants).

    Returns:
        ChatCompletionResponse object or generator of ChatCompletionResponse objects for streaming.

    Raises:
        HTTPException: If the response is invalid or the request fails.
    """
    logger.info(f"Response status code: {response.status_code}")
    logger.info(f"Response headers: {response.headers}")

    # Validate Content-Type
    content_type = response.headers.get("Content-Type", "")
    expected_content_type = "text/event-stream" if is_stream else "application/json"
    if expected_content_type not in content_type:
        logger.error(
            f"Unexpected Content-Type: {content_type}, expected {expected_content_type}")
        raise HTTPException(
            status_code=500, detail=f"Server returned unexpected Content-Type: {content_type}")

    response.raise_for_status()

    # Handle streaming response
    if is_stream:
        def stream_chunks():
            for line in response.iter_lines(decode_unicode=True):
                if not line.strip():
                    continue
                logger.info(f"Stream chunk: {line}")

                # Parse SSE format (expecting "data: {...}")
                if not line.startswith("data: "):
                    logger.error(f"Invalid SSE chunk format: {line}")
                    raise HTTPException(
                        status_code=500, detail="Invalid server-sent event format")

                # Extract JSON data
                json_data = line[len("data: "):].strip()
                if not json_data:
                    logger.error("Empty JSON data in SSE chunk")
                    continue

                try:
                    chunk = json.loads(json_data)
                    # Handle cases where logprobs.tokens is null
                    for choice in chunk.get("choices", []):
                        if choice.get("logprobs") and choice["logprobs"].get("tokens") is None:
                            choice["logprobs"]["tokens"] = []
                    yield ChatCompletionResponse(**chunk)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse chunk JSON: {e}, chunk: {json_data}")
                    raise HTTPException(
                        status_code=500, detail=f"Invalid JSON in streaming chunk: {str(e)}")
        return stream_chunks()

    # Handle non-streaming response
    response_text = response.text
    logger.info(f"Response content: {response_text}")

    if not response_text.strip():
        logger.error("Empty response received from the server")
        raise HTTPException(
            status_code=500, detail="Empty response from MLX LM server")

    try:
        response_json = response.json()
        # Handle cases where logprobs.tokens is null
        for choice in response_json.get("choices", []):
            if choice.get("logprobs") and choice["logprobs"].get("tokens") is None:
                choice["logprobs"]["tokens"] = []
        return ChatCompletionResponse(**response_json)
    except requests.exceptions.JSONDecodeError as e:
        logger.error(
            f"JSON decode error: {str(e)}, Response content: {response_text}")
        raise HTTPException(
            status_code=500, detail=f"Invalid JSON response from server: {str(e)}")

# API Calls


def chat_completions(request: ChatCompletionRequest) -> Union[ChatCompletionResponse, Generator[ChatCompletionResponse, None, None]]:
    """
    Send a chat completion request to the MLX LM server.

    Args:
        request: ChatCompletionRequest object containing the request parameters.

    Returns:
        ChatCompletionResponse object or generator of ChatCompletionResponse objects for streaming.

    Raises:
        HTTPException: If the request fails or the response is invalid.
    """
    try:
        request_payload = request.dict(exclude_none=True)
        logger.info(
            f"Sending request to {BASE_URL}/chat/completions: {request_payload}")

        response = requests.post(
            f"{BASE_URL}/chat/completions",
            json=request_payload,
            headers={"Content-Type": "application/json"},
            stream=request.stream
        )

        return _handle_response(response, is_stream=request.stream, object_type="chat.completion" if not request.stream else "chat.completion.chunk")

    except requests.exceptions.HTTPError as e:
        logger.error(
            f"HTTP error: {str(e)}, Response content: {response.text if 'response' in locals() else 'N/A'}")
        raise HTTPException(
            status_code=500, detail=f"MLX LM server error: {str(e)}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Request to MLX LM server failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal error: {str(e)}")


def text_completions(request: TextCompletionRequest) -> Union[ChatCompletionResponse, Generator[ChatCompletionResponse, None, None]]:
    """
    Send a text completion request to the MLX LM server.

    Args:
        request: TextCompletionRequest object containing the request parameters.

    Returns:
        ChatCompletionResponse object or generator of ChatCompletionResponse objects for streaming.

    Raises:
        HTTPException: If the request fails or the response is invalid.
    """
    try:
        request_payload = request.dict(exclude_none=True)
        logger.info(
            f"Sending request to {BASE_URL}/completions: {request_payload}")

        response = requests.post(
            f"{BASE_URL}/completions",
            json=request_payload,
            headers={"Content-Type": "application/json"},
            stream=request.stream
        )

        return _handle_response(response, is_stream=request.stream, object_type="text.completion" if not request.stream else "text.completion.chunk")

    except requests.exceptions.HTTPError as e:
        logger.error(
            f"HTTP error: {str(e)}, Response content: {response.text if 'response' in locals() else 'N/A'}")
        raise HTTPException(
            status_code=500, detail=f"MLX LM server error: {str(e)}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Request to MLX LM server failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal error: {str(e)}")


def list_models() -> ModelsResponse:
    """
    List available models on the MLX LM server.

    Returns:
        ModelsResponse object containing the list of available models.

    Raises:
        HTTPException: If the request fails.
    """
    try:
        response = requests.get(
            f"{BASE_URL}/models",
            headers={"Content-Type": "application/json"}
        )
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response content: {response.text}")

        content_type = response.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            logger.error(f"Unexpected Content-Type: {content_type}")
            raise HTTPException(
                status_code=500, detail=f"Server returned non-JSON response: Content-Type {content_type}")

        response.raise_for_status()
        return ModelsResponse(**response.json())

    except requests.exceptions.HTTPError as e:
        logger.error(
            f"HTTP error: {str(e)}, Response content: {response.text if 'response' in locals() else 'N/A'}")
        raise HTTPException(
            status_code=500, detail=f"MLX LM server error: {str(e)}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Request to MLX LM server failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Internal error: {str(e)}")
