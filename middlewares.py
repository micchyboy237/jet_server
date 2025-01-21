import traceback
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from jet.utils import get_class_name
from jet.logger import logger


async def log_exceptions_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        logger.error(f"Unhandled exception: {traceback.format_exc()}")
        logger.warning(f"Global: Handled {get_class_name(exc)}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"}
        )
