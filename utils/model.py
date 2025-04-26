import mlx.core as mx
from typing import Union, List, Dict, Optional
from transformers import PreTrainedTokenizer
from mlx_lm.tokenizer_utils import TokenizerWrapper


def get_max_context_length(model: 'nn.Module', max_kv_size: Optional[int] = None) -> int:
    """
    Retrieve the maximum context length of the model (input + output tokens).

    Args:
        model (nn.Module): The MLX model.
        max_kv_size (Optional[int]): The maximum key-value cache size, if specified.

    Returns:
        int: The maximum context length (in tokens).
    """
    # Try to get max context length from model configuration
    try:
        max_context_length = model.config.max_position_embeddings
    except AttributeError:
        # Fallback to a default or max_kv_size if config doesn't specify
        max_context_length = max_kv_size if max_kv_size is not None else 2048  # Default fallback
        print(
            f"Warning: max_position_embeddings not found in model config. Using {max_context_length}.")

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
        # Encode string prompt, adding special tokens if needed
        tokens = tokenizer.encode(
            prompt, add_special_tokens=add_special_tokens)
    elif isinstance(prompt, mx.array):
        tokens = prompt
    else:
        tokens = mx.array(prompt)

    return tokens.size if isinstance(tokens, mx.array) else len(tokens)


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
