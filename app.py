from datetime import datetime
import os
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jet.utils.time_utils import format_time
from routes.vector import router as vector_router
from routes.rag import router as rag_router
from routes.ner import router as ner_router
from routes.prompt import router as prompt_router
from routes.graph import router as graph_router
from routes.job.cover_letter import router as cover_letter_router
from routes.eval.faithfulness import router as faithfulness_router
from middlewares import log_exceptions_middleware
from jet.llm.ollama.base import initialize_ollama_settings
from jet.logger import logger
from contextlib import asynccontextmanager

from shared.setup.events import EventSettings

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Start time tracking
start_time = EventSettings.event_data['pre_start_hook']['start_time']
# start_time = time.time()

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.middleware("http")(auth_memgraph_middleware)

# Middleware to Catch 401 Errors and Retry
# app.add_middleware(AuthMemgraphRetryOn401Middleware)
app.middleware("http")(log_exceptions_middleware)

# Exception Handler for 401 Unauthorized Errors


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# Include routers
app.include_router(rag_router, prefix="/api/v1/rag", tags=["rag"])
app.include_router(vector_router, prefix="/api/v1/vector", tags=["vector"])
app.include_router(ner_router, prefix="/api/v1/ner", tags=["ner"])
app.include_router(prompt_router, prefix="/api/v1/prompt", tags=["prompt"])
app.include_router(graph_router, prefix="/api/v1/graph", tags=["graph"])
app.include_router(cover_letter_router,
                   prefix="/api/v1/job/cover-letter", tags=["job", "cover-letter"])
app.include_router(faithfulness_router,
                   prefix="/api/v1/eval/faithfulness", tags=["evaluation", "faithfulness"])

# Lifespan event manager with startup duration logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize resources before the app starts
    initialize_ollama_settings()

    # # End the timer (time.time() is already a timestamp)
    # end_time = time.time()

    # # Calculate the duration (both are in float timestamp format)
    # startup_duration = end_time - start_time

    # # Format the duration
    # formatted_duration = format_time(startup_duration)

    # logger.log("Startup duration:", formatted_duration,
    #            colors=["GRAY", "SUCCESS"])

    yield


app = FastAPI(lifespan=lifespan)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        reload_dirs=[
            "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules"
        ],
        # reload_excludes=["**/.venv/**"],  # Exclude .venv from being watched
    )
