from mlx_lm import stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import PreTrainedTokenizer
from typing import Union, List, Dict, Optional
import mlx.core as mx


def get_max_context_length(
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    max_kv_size: Optional[int] = None
) -> int:
    """
    Retrieve the maximum context length using the tokenizer's configuration.

    Args:
        tokenizer (Union[PreTrainedTokenizer, TokenizerWrapper]): The tokenizer.
        max_kv_size (Optional[int]): The maximum key-value cache size, if specified.

    Returns:
        int: The maximum context length (in tokens).
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    # Try to get max context length from tokenizer configuration
    max_context_length = None
    try:
        max_context_length = tokenizer.model_max_length
        if not isinstance(max_context_length, int) or max_context_length <= 0:
            max_context_length = None
    except AttributeError:
        pass

    # Fall back to max_kv_size or default if tokenizer doesn't provide a valid value
    if max_context_length is None:
        max_context_length = max_kv_size if max_kv_size is not None else 2048
        print(
            f"Warning: tokenizer.model_max_length not found. Using default: {max_context_length}.")

    # If max_kv_size is specified and smaller, it limits the context length
    if max_kv_size is not None and max_kv_size < max_context_length:
        max_context_length = max_kv_size
        print(
            f"Max context length limited by max_kv_size: {max_context_length}")

    return max_context_length


def get_prompt_token_count(
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, mx.array, List[int]],
    add_special_tokens: bool = True
) -> int:
    """
    Calculate the token count for a given prompt.

    Args:
        tokenizer (Union[PreTrainedTokenizer, TokenizerWrapper]): The tokenizer.
        prompt (Union[str, mx.array, List[int]]): The input prompt (string, token array, or token list).
        add_special_tokens (bool): Whether to add special tokens (e.g., BOS) during encoding.

    Returns:
        int: The number of tokens in the prompt.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if isinstance(prompt, str):
        tokens = tokenizer.encode(
            prompt, add_special_tokens=add_special_tokens)
    elif isinstance(prompt, mx.array):
        tokens = prompt
    else:
        tokens = mx.array(prompt)

    return tokens.size if isinstance(tokens, mx.array) else len(tokens)


def get_response_token_count(
    model: 'nn.Module',
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: Union[str, mx.array, List[int]],
    max_tokens: int = 100,
    **kwargs
) -> tuple[str, int]:
    """
    Calculate the token count for the generated response.

    Args:
        model (nn.Module): The MLX model.
        tokenizer (Union[PreTrainedTokenizer, TokenizerWrapper]): The tokenizer.
        prompt (Union[str, mx.array, List[int]]): The input prompt.
        max_tokens (int): Maximum number of tokens to generate.
        **kwargs: Additional arguments passed to stream_generate (e.g., sampler, draft_model).

    Returns:
        tuple[str, int]: The generated text and the number of tokens in the response.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    text = ""
    response_token_count = 0

    for response in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, **kwargs):
        text += response.text
        response_token_count = response.generation_tokens
        # Stop if generation is complete (e.g., EOS or max_tokens reached)
        if response.finish_reason in ["stop", "length"]:
            break

    return text, response_token_count


def get_messages_token_count(
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    messages: List[Dict[str, str]],
    chat_template_config: Optional[Dict] = None,
    add_special_tokens: bool = False,
    continue_final_message: bool = False,
    add_generation_prompt: bool = True
) -> int:
    """
    Calculate the token count for a list of messages, applying the chat template if available.

    Args:
        tokenizer (Union[PreTrainedTokenizer, TokenizerWrapper]): The tokenizer.
        messages (List[Dict[str, str]]): List of messages with 'role' and 'content' keys.
        chat_template_config (Optional[Dict]): Additional config for chat template.
        add_special_tokens (bool): Whether to add special tokens during encoding.
        continue_final_message (bool): Whether to continue the final message (for prefill).
        add_generation_prompt (bool): Whether to add a generation prompt.

    Returns:
        int: The total number of tokens for the messages.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    chat_template_config = chat_template_config or {}

    # Apply chat template if available
    if tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=continue_final_message,
            add_generation_prompt=add_generation_prompt,
            **chat_template_config
        )
        tokens = tokenizer.encode(
            prompt, add_special_tokens=add_special_tokens)
    else:
        # Concatenate message contents and encode
        prompt = "".join(message["content"] for message in messages)
        tokens = tokenizer.encode(
            prompt, add_special_tokens=add_special_tokens)

    return tokens.size if isinstance(tokens, mx.array) else len(tokens)


def get_individual_message_token_counts(
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    messages: List[Dict[str, str]],
    add_special_tokens: bool = False
) -> List[Dict[str, Union[str, int]]]:
    """
    Calculate the token count for each message individually.

    Args:
        tokenizer (Union[PreTrainedTokenizer, TokenizerWrapper]): The tokenizer.
        messages (List[Dict[str, str]]): List of messages with 'role' and 'content' keys.
        add_special_tokens (bool): Whether to add special tokens during encoding.

    Returns:
        List[Dict[str, Union[str, int]]]: List of dictionaries with 'role', 'content', and 'token_count'.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    result = []
    for message in messages:
        tokens = tokenizer.encode(
            message["content"], add_special_tokens=add_special_tokens)
        token_count = tokens.size if isinstance(
            tokens, mx.array) else len(tokens)
        result.append({
            "role": message["role"],
            "content": message["content"],
            "token_count": token_count
        })
    return result
