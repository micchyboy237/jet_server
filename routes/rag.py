import json
import time
from typing import Any, Awaitable, Generator, Literal, Optional
from jet.data.utils import generate_key
from jet.llm.ollama.constants import OLLAMA_LARGE_EMBED_MODEL
from jet.llm.utils.embeddings import get_ollama_embedding_function
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
from jet.memory.lru_cache import LRUCache
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore, TextNode
import requests
from tqdm import tqdm
from jet.actions.prompts_generator import PromptsGenerator
from jet.llm.ollama.base import Ollama
from jet.transformers.formatters import format_json
from pydantic import BaseModel
from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from jet.vectors import get_source_node_attributes
from jet.logger import logger

from helpers.rag import RAG
from config import stop_event

router = APIRouter()

# Create default RAG instance (will be updated in the endpoint)
rag_global_dict = LRUCache(max_size=5)

rag_dir: str = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
json_attributes: list[str] = ["title", "details"]
exclude_json_attributes: list[str] = []
metadata_attributes: list[str] = []
extensions: list[str] = [".md", ".mdx", ".rst"]
system: str = (
    "You are a job applicant providing tailored responses during an interview.\n"
    "Always answer questions using the provided context as if it is your resume, "
    "and avoid referencing the context directly.\n"
    "Some rules to follow:\n"
    "1. Never directly mention the context or say 'According to my resume' or similar phrases.\n"
    "2. Provide responses as if you are the individual described in the context, focusing on professionalism and relevance."
)
chunk_size: Optional[int] = None
chunk_overlap: int = 40
sub_chunk_sizes: list[int] = [512, 256, 128]
with_hierarchy: bool = True
top_k: Optional[int] = None
model: str = "llama3.2"
embed_model: str = OLLAMA_LARGE_EMBED_MODEL
mode: Literal["annoy", "fusion", "bm25", "hierarchy",
              "deeplake", "faiss", "graph_nx"] = "fusion"
store_path: str = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_server/.cache/deeplake/store_1"
score_threshold: float = 0.0
split_mode: list[Literal["markdown", "hierarchy"]] = []
fusion_mode: FUSION_MODES = FUSION_MODES.SIMPLE
contexts: list[str] = []
disable_chunking: Optional[bool] = False


# Define the schema for input queries
class QueryRequest(BaseModel):
    query: str
    rag_dir: str = rag_dir
    extensions: list[str] = extensions
    json_attributes: list[str] = json_attributes
    exclude_json_attributes: list[str] = exclude_json_attributes
    metadata_attributes: list[str] = metadata_attributes
    system: str = system
    chunk_size: Optional[int] = chunk_size
    chunk_overlap: int = chunk_overlap
    sub_chunk_sizes: list[int] = sub_chunk_sizes
    with_hierarchy: bool = with_hierarchy
    top_k: Optional[int] = top_k
    model: str = model
    embed_model: str = embed_model
    mode: Literal["annoy", "fusion", "bm25", "hierarchy",
                  "deeplake", "faiss", "graph_nx"] = mode
    store_path: str = store_path
    score_threshold: float = score_threshold
    split_mode: list[Literal["markdown", "hierarchy"]] = split_mode
    fusion_mode: FUSION_MODES = fusion_mode
    disable_chunking: Optional[bool] = False


class SearchRequest(QueryRequest):
    contexts: list[str] = contexts


class VectorNode(BaseModel):
    id: str
    score: float
    text: str
    metadata: Optional[dict] = None


class VectorNodesResponse(BaseModel):
    count: int  # Add count to track the number of nodes
    data: list[VectorNode]

    @classmethod
    def from_nodes(cls, nodes: list[dict]):
        """Transform the nodes, changing 'node_id' to 'id'."""
        # Transform nodes
        transformed_nodes = [
            {
                **get_source_node_attributes(node),
                "id": get_source_node_attributes(node).pop("node_id"),
            }
            for node in nodes
        ]
        # Return the response with both count and data
        return cls(count=len(transformed_nodes), data=transformed_nodes)


def setup_rag(**kwargs) -> RAG:
    """
    Setup a RAG object and store it in a global dictionary with a unique mode.

    Args:
        rag_dir (str): Path to the RAG directory.
        **kwargs: Additional arguments for RAG initialization.

    Returns:
        RAG: The initialized RAG object.
    """
    global rag_global_dict

    mode = kwargs.get("mode")
    if mode is None:
        raise ValueError("The 'mode' key must be provided in kwargs.")

    # Dependencies for hash computation
    deps = ["path_or_docs", "json_attributes", "exclude_json_attributes", "metadata_attributes",
            "chunk_size", "chunk_overlap", "disable_chunking"]
    deps_values = [kwargs[key] for key in deps if key in kwargs]
    current_hash = generate_key(*deps_values)

    # Check if cache needs to be cleared due to changed hash
    last_hash = rag_global_dict.get("last_hash")
    if last_hash and last_hash != current_hash:
        logger.warning("Detected change in dependencies. Clearing RAG cache.")
        rag_global_dict.clear()

    # Check if RAG with the same mode and hash already exists
    existing_entry = rag_global_dict.get(current_hash)
    if existing_entry:
        logger.info("Reusing existing RAG object with mode: %s", mode)
        return existing_entry["rag"]

    # Initialize the RAG object
    rag = RAG(**kwargs)

    # Store the latest hash for comparison
    rag_global_dict.put("last_hash", current_hash)

    # Cache the new RAG object
    rag_global_dict.put(current_hash, {
        "rag": rag,
        "mode": mode,
        "hash": current_hash
    })

    logger.debug("Created RAG object for cache key: %s", current_hash)
    logger.info("Cached RAG in memory: %d", len(rag_global_dict))
    logger.debug(format_json({key: value if not isinstance(
        value, dict) else value['mode'] for key, value in rag_global_dict.items()}))

    return rag


def generate_sub_prompts(prompts: list[str]) -> Generator[str, None, None]:
    """Generator function to yield events for streaming."""
    processor = PromptsGenerator(llm=Ollama(model="llama3.1"))
    response_stream = processor.process(prompts)

    generation_tqdm = tqdm(response_stream)

    for (text, result) in generation_tqdm:
        yield result
        logger.success(result)

        # Update the progress bar after processing each node
        generation_tqdm.update(1)


# @router.post("/query")
# async def query(request: SearchRequest):

#     headers = {
#         "Cache-Control": "no-cache",
#         "Connection": "keep-alive",
#         "Content-Type": "text/event-stream",
#     }
#     return StreamingResponse(event_stream_query(request), headers=headers)


@router.get("/query")
async def query(
    query: str = Query(...),
    rag_dir: str = Query(default=rag_dir),
    extensions: list[str] = Query(default=extensions),
    system: str = Query(default=system),
    chunk_size: int = Query(default=chunk_size),
    chunk_overlap: int = Query(default=chunk_overlap),
    sub_chunk_sizes: list[int] = Query(default=sub_chunk_sizes),
    with_hierarchy: bool = Query(default=with_hierarchy),
    top_k: Optional[int] = Query(default=top_k),
    model: str = Query(default=model),
    embed_model: str = Query(default=embed_model),
    mode: Literal["annoy", "fusion", "bm25", "hierarchy",
                  "deeplake", "faiss", "graph_nx"] = Query(default=mode),
    store_path: str = Query(default=store_path),
    score_threshold: float = Query(default=score_threshold),
    split_mode: list[Literal["markdown", "hierarchy"]
                     ] = Query(default=split_mode),
    contexts: list[str] = Query(default=contexts),
):
    global stop_event

    if stop_event.is_set():
        stop_event.clear()

    headers = {
        "Cache-Control": "no-cache",
        # "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
    }
    search_request = SearchRequest(
        query=query,
        rag_dir=rag_dir,
        extensions=extensions,
        json_attributes=json_attributes,
        exclude_json_attributes=exclude_json_attributes,
        metadata_attributes=metadata_attributes,
        system=system,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        sub_chunk_sizes=sub_chunk_sizes,
        with_hierarchy=with_hierarchy,
        top_k=top_k,
        model=model,
        embed_model=embed_model,
        mode=mode,
        store_path=store_path,
        score_threshold=score_threshold,
        split_mode=split_mode,
        fusion_mode=fusion_mode,
        contexts=contexts,
    )
    return StreamingResponse(event_stream_query(search_request), headers=headers)


def event_stream_query(search_request: SearchRequest):
    search_request_dict = search_request.__dict__.copy()
    query = search_request_dict.pop("query")
    system = search_request_dict.pop("system")
    contexts = search_request_dict.pop("contexts")
    top_k = search_request.top_k

    rag = setup_rag(
        system=system,
        path_or_docs=search_request_dict.pop("rag_dir"),
        **search_request_dict
    )

    for chunk in rag.query(query, contexts, top_k=top_k, system=system, stop_event=stop_event):
        message = f"data: {chunk}\n\n"
        yield message


@router.post("/query/stop")
async def query_stop():
    global stop_event
    # Start streaming in a thread
    # thread = threading.Thread(target=stream_chat)
    # thread.start()

    stop_event.set()

    time.sleep(1)
    response = requests.post("http://localhost:11434/api/chat/stop")

    # thread.join()
    logger.newline()
    logger.purple("Stream fully stopped.")
    logger.purple("Response:", response)


# @router.post("/nodes", response_model=VectorNodesResponse)
@router.post("/nodes")
async def get_nodes(query_request: QueryRequest):
    query_request_dict = query_request.__dict__.copy()
    query = query_request_dict.pop("query")

    rag = setup_rag(
        path_or_docs=query_request_dict.pop("rag_dir"),
        **query_request_dict
    )

    result = rag.get_results(query, **query_request_dict)

    # data = VectorNodesResponse.from_nodes(result["nodes"])
    data = result["nodes"]

    return {
        "data": data
    }


@router.post("/stream-nodes", response_model=VectorNodesResponse)
async def stream_nodes(query_request: QueryRequest):
    headers = {
        "Cache-Control": "no-cache",
        # "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
    }
    return StreamingResponse(event_stream_nodes(query_request), headers=headers)


def event_stream_nodes(query_request: QueryRequest) -> Generator[str, None, None]:
    """Generator function to yield events for streaming."""
    query_request_dict = query_request.__dict__.copy()
    query = query_request_dict.pop("query")
    top_k = query_request.top_k

    rag = setup_rag(
        path_or_docs=query_request_dict.pop("rag_dir"),
        **query_request_dict
    )

    stream_prompts = generate_sub_prompts([query])

    for prompt in stream_prompts:
        result = rag.get_results(
            prompt, top_k=top_k)

        transformed_nodes = VectorNodesResponse.from_nodes(result["nodes"])
        yield f"data: {transformed_nodes}\n\n"

        logger.debug("Result Prompt:", prompt)
        logger.success(format_json(transformed_nodes))


@router.post("/sample-stream")
async def sample_stream_post(request: Request):
    """
    Endpoint to stream responses for a given thread_id.
    """

    body = await request.body()
    request_params_str = body.decode('utf-8')
    request_params = json.loads(request_params_str)

    logger.debug("POST sample_stream:")
    logger.orange(format_json(request_params))

    headers = {
        "Cache-Control": "no-cache",
        # "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
    }
    return StreamingResponse(event_sample_stream(request_params), headers=headers)


@router.get("/sample-stream")
async def sample_stream_get():
    """
    Endpoint to stream responses for a given thread_id.
    """

    logger.debug("GET sample_stream:")

    headers = {
        "Cache-Control": "no-cache",
        # "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
    }
    return StreamingResponse(event_sample_stream(), headers=headers)


def event_sample_stream(request_params: Optional[Any] = None):
    """Generator function to yield events for streaming."""
    for i in range(10):  # Example: Stream 10 chunks
        chunk = f"data: Message {i}\n\n"
        yield chunk
        logger.success(chunk)
        time.sleep(1)  # Simulate delay between chunks


if __name__ == "__main__":
    import asyncio
    from pprint import pprint

    async def main():
        query = "Tell me about yourself."
        # Create a QueryRequest object with the specified parameters
        query_request = QueryRequest(
            query=query,
            chunk_size=1024,
            chunk_overlap=40,
            sub_chunk_sizes=[512, 256, 128],
            top_k=20,
            mode="hierarchy",
        )

        # Call the get_nodes endpoint
        response = await get_nodes(query_request)

        # Print the response
        logger.debug(query)
        logger.success(format_json(response.data))

    asyncio.run(main())
