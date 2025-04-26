import asyncio
import time
import gc
import mlx.core as mx
from fastapi import HTTPException
from jet.logger import logger
from models_config import AVAILABLE_MODELS
from mlx_lm import load

# Shared model cache
MODEL_CACHE = {
    "model": None,
    "tokenizer": None,
    "model_name": None,
    "last_used": None
}
MODEL_CACHE_DURATION = 180  # 3 minutes
MODEL_CACHE_LOCK = asyncio.Lock()


def unload_current_model():
    """Unload the currently cached model to free RAM."""
    if MODEL_CACHE["model"] is not None:
        logger.info(f"Unloading model: {MODEL_CACHE['model_name']}")
        MODEL_CACHE["model"] = None
        MODEL_CACHE["tokenizer"] = None
        MODEL_CACHE["model_name"] = None
        MODEL_CACHE["last_used"] = None
        mx.clear_cache()
        gc.collect()
        logger.info("Model unloaded and memory cleared.")


async def load_model(model_name: str):
    """Load a model and update the cache, unloading the current model if necessary."""
    async with MODEL_CACHE_LOCK:
        if model_name != MODEL_CACHE["model_name"]:
            unload_current_model()
            if model_name not in AVAILABLE_MODELS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid model. Available models: {list(AVAILABLE_MODELS.keys())}"
                )
            model_path = AVAILABLE_MODELS[model_name]
            logger.info(f"Loading model: {model_path}")
            model, tokenizer = load(model_path)
            MODEL_CACHE["model"] = model
            MODEL_CACHE["tokenizer"] = tokenizer
            MODEL_CACHE["model_name"] = model_name
            MODEL_CACHE["last_used"] = time.time()
            logger.info(f"Model {model_name} loaded and cached.")
        else:
            MODEL_CACHE["last_used"] = time.time()
        return MODEL_CACHE["model"], MODEL_CACHE["tokenizer"]


async def cleanup_idle_models():
    """Background task to unload models idle for more than MODEL_CACHE_DURATION."""
    while True:
        async with MODEL_CACHE_LOCK:
            if MODEL_CACHE["model"] is not None and MODEL_CACHE["last_used"] is not None:
                idle_time = time.time() - MODEL_CACHE["last_used"]
                if idle_time > MODEL_CACHE_DURATION:
                    logger.info(
                        f"Model {MODEL_CACHE['model_name']} idle for {idle_time:.2f} seconds, unloading.")
                    unload_current_model()
        await asyncio.sleep(10)
