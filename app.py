import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.vector import router as vector_router
from routes.rag import router as rag_router
from routes.ner import router as ner_router
from routes.prompt import router as prompt_router
from middlewares import log_exceptions_middleware
from jet.llm.ollama.base import initialize_ollama_settings
initialize_ollama_settings()

# Enable parallelism for faster LLM tokenizer encoding
os.environ["TOKENIZERS_PARALLELISM"] = "true"


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.middleware("http")(log_exceptions_middleware)

# Include the routes
app.include_router(rag_router, prefix="/api/v1/rag", tags=["rag"])
app.include_router(vector_router, prefix="/api/v1/vector", tags=["vector"])
app.include_router(ner_router, prefix="/api/v1/ner", tags=["ner"])
app.include_router(prompt_router, prefix="/api/v1/prompt", tags=["prompt"])


# Run a simple test if this module is the main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8002, reload=True, reload_dirs=[
        "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules"
    ])
