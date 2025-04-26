# Configuration file for available MLX models with shortened names
from jet.transformers.formatters import format_json
from jet.utils.object import max_getattr
from jet.logger import logger
from transformers import AutoConfig

AVAILABLE_MODELS = {
    "deepseek-r1-14b": "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
    "dolphin-llama3.1-8b": "mlx-community/Dolphin3.0-Llama3.1-8B-4bit",
    "dolphin-llama3.2-3b": "mlx-community/dolphin3.0-llama3.2-3B-4Bit",
    "gemma3-1b": "mlx-community/gemma-3-1b-it-qat-4bit",
    "gemma3-4b": "mlx-community/gemma-3-4b-it-qat-4bit",
    "gemma3-12b": "mlx-community/gemma-3-12b-it-qat-4bit",
    "llama3.1-8b": "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "llama3.2-3b": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mistral-nemo": "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
    "phi3.5-mini": "mlx-community/Phi-3.5-mini-instruct-4bit",
    "phi4": "mlx-community/phi-4-4bit",
    "qwen2.5-7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "qwen2.5-14b": "mlx-community/Qwen2.5-14B-Instruct-4bit",
    "qwen2.5-coder-14b": "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"
}

MODEL_CONTEXTS = {
    "deepseek-r1-14b": 131072,
    "dolphin-llama3.1-8b": 131072,
    "dolphin-llama3.2-3b": 131072,
    "gemma3-1b": 32768,
    "gemma3-4b": 131072,
    "gemma3-12b": 131072,
    "llama3.1-8b": 131072,
    "llama3.2-3b": 131072,
    "mistral-nemo": 1024000,
    "phi3.5-mini": 131072,
    "phi4": 16384,
    "qwen2.5-7b": 32768,
    "qwen2.5-14b": 32768,
    "qwen2.5-coder-14b": 32768
}

MODEL_EMBEDDING_TOKENS = {
    "deepseek-r1-14b": 5120,
    "dolphin-llama3.1-8b": 4096,
    "dolphin-llama3.2-3b": 3072,
    "gemma3-1b": 1152,
    "gemma3-4b": 2560,
    "gemma3-12b": 3840,
    "llama3.1-8b": 4096,
    "llama3.2-3b": 3072,
    "mistral-nemo": 5120,
    "phi3.5-mini": 3072,
    "phi4": 5120,
    "qwen2.5-7b": 3584,
    "qwen2.5-14b": 5120,
    "qwen2.5-coder-14b": 5120
}


def get_model_limits(model_id):
    config = AutoConfig.from_pretrained(model_id)

    max_context = max_getattr(config, 'max_position_embeddings', None)
    # or `config.hidden_dim`
    max_embeddings = max_getattr(config, 'hidden_size', None)

    return max_context, max_embeddings


def get_model_info():
    model_info = {"contexts": {}, "embeddings": {}}
    for short_name, model_path in AVAILABLE_MODELS.items():
        try:
            max_contexts, max_embeddings = get_model_limits(model_path)
            if not max_contexts:
                raise ValueError(
                    f"Missing 'max_position_embeddings' from {model_path} config")
            elif not max_embeddings:
                raise ValueError(
                    f"Missing 'hidden_size' from {model_path} config")

            print(
                f"{short_name}: max_contexts={max_contexts}, max_embeddings={max_embeddings}")

            model_info["contexts"][short_name] = max_contexts
            model_info["embeddings"][short_name] = max_embeddings

        except Exception as e:
            logger.error(f"Failed to get config for {short_name}: {e}")
            raise

    return model_info
