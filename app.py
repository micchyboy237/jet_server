import os
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routes.vector import router as vector_router
from routes.rag import router as rag_router
from routes.ner import router as ner_router
from routes.prompt import router as prompt_router
from routes.graph import router as graph_router
from routes.job.cover_letter import router as cover_letter_router
from middlewares import log_exceptions_middleware, AuthMemgraphRetryOn401Middleware
from jet.llm.ollama.base import initialize_ollama_settings
from jet.logger import logger

from starlette.middleware.base import BaseHTTPMiddleware

initialize_ollama_settings()
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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
app.add_middleware(AuthMemgraphRetryOn401Middleware)
# app.middleware("http")(log_exceptions_middleware)


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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8002, reload=True, reload_dirs=[
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules"
    ])
    # uvicorn.run("app:app", host="0.0.0.0", port=8002)
