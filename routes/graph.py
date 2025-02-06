import httpx
from typing import TypedDict
from typing import Generator
from fastapi import APIRouter, HTTPException, FastAPI, Request, Depends
from pydantic import BaseModel
from jet.memory.memgraph import generate_query, generate_cypher_query, initialize_graph
from jet.logger import logger
from starlette.responses import StreamingResponse
from tqdm import tqdm
import os

from jet.memory.memgraph import (
    authenticate_user,
    query_memgraph,
)

from jet.memory.memgraph_types import (
    AuthResponse,
    GraphQueryResponse,
    LoginRequest,
    CypherQueryRequest,
    GraphQueryMetadata,
    GraphQueryRequest
)

router = APIRouter()

# Environment variables for Memgraph connection
MEMGRAPH_URI = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USERNAME = os.environ.get("MEMGRAPH_USERNAME", "")
MEMGRAPH_PASSWORD = os.environ.get("MEMGRAPH_PASSWORD", "")

graph = initialize_graph(MEMGRAPH_URI, MEMGRAPH_USERNAME, MEMGRAPH_PASSWORD)


router = APIRouter()


@router.post("/login", response_model=AuthResponse)
async def login(request: LoginRequest):
    return {
        "data": authenticate_user(request.dict())
    }


def event_cypher_query_stream(request: CypherQueryRequest) -> Generator[str, None, None]:
    """Generator function to yield Cypher queries for streaming."""
    try:
        generated_cypher_queries = generate_cypher_query(
            request.query, graph, request.tone_name, num_of_queries=request.num_of_queries
        )
        generation_tqdm = tqdm(generated_cypher_queries)

        for query in generation_tqdm:
            yield f"data: {query}\n\n"
            logger.success(query)
            generation_tqdm.update(1)
    except Exception as e:
        logger.error(f"Error generating Cypher queries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-cypher-queries", response_model=list[str])
async def generate_cypher_queries(request: CypherQueryRequest):
    """
    Generate Cypher queries based on the given user query.
    """
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
    }
    return StreamingResponse(event_cypher_query_stream(request), headers=headers)


@router.post("/query-graph")
async def query_graph(request: GraphQueryRequest):
    """
    Query the knowledge graph and return formatted results.
    """

    graph_result = query_memgraph(request)

    return {
        "data": graph_result
    }


@router.post("/query-graph-old")
async def query_graph(request: GraphQueryRequest):
    """
    Query the knowledge graph and return formatted results.
    """

    graph_result = graph.query(request.query)

    return {
        "data": graph_result
    }
