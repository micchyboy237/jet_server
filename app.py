import os
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routes.rerankers.heuristic import router as reranker_heuristic_router
from routes.rerankers.semantic import router as reranker_semantic_router
from routes.rag import router as rag_router
from routes.ner import router as ner_router
from routes.prompt import router as prompt_router
from routes.search import router as search_router
from routes.graph import router as graph_router
from routes.job.cover_letter import router as cover_letter_router
from routes.eval.faithfulness import router as faithfulness_router
from routes.evaluation import router as evaluation_router
# Import cleanup_idle_models
from routes.text_generation import router as text_generation_router, cleanup_idle_models
from middlewares import log_exceptions_middleware
from jet.llm.ollama.base import initialize_ollama_settings
from jet.logger import logger
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio

initialize_ollama_settings()
os.environ["TOKENIZERS_PARALLELISM"] = "true"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(log_exceptions_middleware)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

# Startup event to start the cleanup_idle_models task


@app.on_event("startup")
async def startup_event():
    logger.info("Starting cleanup_idle_models task")
    asyncio.create_task(cleanup_idle_models())

app.include_router(rag_router, prefix="/api/v1/rag", tags=["rag"])
app.include_router(reranker_heuristic_router,
                   prefix="/api/v1/reranker/heuristic", tags=["reranker", "heuristic"])
app.include_router(reranker_semantic_router,
                   prefix="/api/v1/reranker/semantic", tags=["reranker", "semantic"])
app.include_router(ner_router, prefix="/api/v1/ner", tags=["ner"])
app.include_router(prompt_router, prefix="/api/v1/prompt", tags=["prompt"])
app.include_router(search_router, prefix="/api/v1/search", tags=["search"])
app.include_router(graph_router, prefix="/api/v1/graph", tags=["graph"])
app.include_router(cover_letter_router,
                   prefix="/api/v1/job/cover-letter", tags=["job", "cover-letter"])
app.include_router(evaluation_router,
                   prefix="/api/v1/evaluation", tags=["evaluation", "models"])
app.include_router(faithfulness_router,
                   prefix="/api/v1/eval/faithfulness", tags=["evaluation", "faithfulness"])
app.include_router(text_generation_router,
                   prefix="/api/v1/text-generation", tags=["text-generation"])


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down, cancelling cleanup_idle_models task")
    tasks = [task for task in asyncio.all_tasks(
    ) if task is not asyncio.current_task()]
    for task in tasks:
        task.cancel()

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
    )
