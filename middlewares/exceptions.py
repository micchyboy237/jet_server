import json
import traceback
from fastapi import Request
from jet.transformers.formatters import format_json
from jet.transformers.object import make_serializable
from jet.utils.inspect_utils import log_filtered_stack_trace
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from jet.utils.class_utils import get_class_name
from jet.logger import logger


async def log_exceptions_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        error = make_serializable(exc)

        # Log general error info
        logger.error(f"Exception: {format_json(error)}")
        logger.gray(traceback.format_exc())

        # Log filtered stack trace
        log_filtered_stack_trace(exc)

        logger.warning(f"Global: Handled {get_class_name(exc)}")

        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "error": error}
        )
