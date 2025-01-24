import hashlib
import json
from typing import Any, Awaitable, Generator, Literal, Optional
from deeplake.core.vectorstore.deeplake_vectorstore import VectorStore
from jet.llm.ollama.embeddings import get_ollama_embedding_function
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
from llama_index.core.schema import NodeWithScore, TextNode
from tqdm import tqdm
from jet.llm.main.prompts_generator import PromptsGenerator
from jet.llm.ollama.base import Ollama
from jet.transformers.formatters import format_json
from pydantic import BaseModel
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from jet.vectors import get_source_node_attributes
from jet.logger import logger

from helpers.rag import RAG

router = APIRouter()

# Create default RAG instance (will be updated in the endpoint)
rag_global_dict: dict[str, object] = {}


# Define the schema for input queries
class QueryRequest(BaseModel):
    query: str
    rag_dir: str = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
    extensions: list[str] = [".md", ".mdx", ".rst"]
    system: str = (
        "You are a job applicant providing tailored responses during an interview.\n"
        "Always answer questions using the provided context as if it is your resume, "
        "and avoid referencing the context directly.\n"
        "Some rules to follow:\n"
        "1. Never directly mention the context or say 'According to my resume' or similar phrases.\n"
        "2. Provide responses as if you are the individual described in the context, focusing on professionalism and relevance."
    )
    chunk_size: int = 1024
    chunk_overlap: int = 40
    sub_chunk_sizes: list[int] = []
    with_hierarchy: bool = True
    top_k: Optional[int] = None
    model: str = "llama3.2"
    embed_model: str = "mxbai-embed-large"
    mode: Literal["fusion", "hierarchy", "deeplake"] = "fusion"
    store_path: str = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_server/.cache/deeplake/store_1"
    score_threshold: float = 0.0


class Metadata(BaseModel):
    file_name: str
    file_path: str
    file_type: str
    file_size: int
    creation_date: str
    last_modified_date: str
    chunk_size: Optional[int] = None
    depth: Optional[int] = None


class Node(BaseModel):
    id: str
    score: float
    text_length: int
    start_end: list[int]
    text: str
    metadata: Metadata


class NodesResponse(BaseModel):
    data: list[Node]

    @classmethod
    def from_nodes(cls, nodes: list):
        # Transform the nodes, changing 'node_id' to 'id'
        transformed_nodes = [
            {
                **get_source_node_attributes(node),
                "id": get_source_node_attributes(node).pop("node_id"),
            }
            for node in nodes
        ]
        return cls(data=transformed_nodes)


def generate_key(*args: Any) -> str:
    """
    Generate a SHA256 hash key from the concatenation of input arguments.

    Args:
        *args: Variable length argument list.

    Returns:
        A SHA256 hash string.
    """
    try:
        # Combine the arguments into a JSON string
        concatenated = json.dumps(args, separators=(',', ':'))
        # Generate a SHA256 hash of the concatenated string
        key = hashlib.sha256(concatenated.encode()).hexdigest()
        return key
    except TypeError as e:
        raise ValueError(f"Invalid argument provided: {e}")


def setup_rag(rag_dir: str, **kwargs):
    """
    Setup a RAG object and store it in a global dictionary with a unique mode.

    Args:
        rag_dir (str): Path to the RAG directory.
        **kwargs: Additional arguments for RAG initialization.

    Returns:
        RAG: The initialized RAG object.
    """
    global rag_global_dict

    # Validate mode key
    mode = kwargs.get("mode")
    if mode is None:
        raise ValueError("The 'mode' key must be provided in kwargs.")

    # Generate hash for the current set of arguments
    deps = [
        "system",
        "chunk_size",
        "chunk_overlap",
        "sub_chunk_sizes",
        "with_hierarchy",
        "embed_model",
        "mode",
        "store_path",
    ]
    deps_values = [kwargs[key] for key in deps if key in kwargs]
    current_hash = generate_key(*deps_values)

    # Check if RAG with the same mode and hash already exists
    existing_key = None
    for key, value in rag_global_dict.items():
        # If mode matches and the hash of parameters is the same
        if value.get("mode") == mode and value.get("hash") == current_hash:
            existing_key = key
            break

    if existing_key:
        # If RAG object exists with the same mode and parameters, reuse it
        logger.info("Reusing existing RAG object with mode: %s", mode)
        return rag_global_dict[existing_key]["rag"]

    # Initialize the RAG object
    rag = RAG(
        path_or_docs=rag_dir,
        **kwargs
    )

    # Cache the new RAG object with the computed hash
    rag_global_dict[current_hash] = {
        "rag": rag,
        "mode": mode,
        "hash": current_hash
    }
    logger.debug("Created RAG object for cache key: %s", current_hash)

    logger.newline()
    logger.info("Cached RAG in memory:", len(rag_global_dict))
    logger.debug([{"mode": value["mode"], "id": key}
                 for key, value in rag_global_dict.items()])
    logger.newline()

    return rag_global_dict[current_hash]["rag"]


def event_stream_query(query_request: QueryRequest):
    query_request_dict = query_request.__dict__.copy()
    query = query_request_dict.pop("query")
    top_k = query_request.top_k

    rag = setup_rag(
        **query_request_dict
    )

    for chunk in rag.query(query, top_k=top_k):
        yield f"data: {chunk}\n\n"


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


@router.post("/query")
async def query(query_request: QueryRequest):

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
    }
    return StreamingResponse(event_stream_query(query_request), headers=headers)


@router.post("/nodes", response_model=NodesResponse)
async def get_nodes(query_request: QueryRequest):
    query_request_dict = query_request.__dict__.copy()
    query = query_request_dict.pop("query")
    top_k = query_request.top_k

    rag = setup_rag(
        **query_request_dict
    )

    result = rag.get_results(query, top_k=top_k)

    data = NodesResponse.from_nodes(result["nodes"])

    return data


def event_stream_nodes(query_request: QueryRequest) -> Generator[str, None, None]:
    """Generator function to yield events for streaming."""
    query_request_dict = query_request.__dict__.copy()
    query = query_request_dict.pop("query")
    top_k = query_request.top_k

    rag = setup_rag(
        **query_request_dict
    )

    stream_prompts = generate_sub_prompts([query])

    for prompt in stream_prompts:
        result = rag.get_results(
            prompt, top_k=top_k)

        transformed_nodes = NodesResponse.from_nodes(result["nodes"])
        yield f"data: {transformed_nodes}\n\n"

        logger.debug("Result Prompt:", prompt)
        logger.success(format_json(transformed_nodes))


@router.post("/stream-nodes", response_model=NodesResponse)
async def stream_nodes(query_request: QueryRequest):
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
    }
    return StreamingResponse(event_stream_nodes(query_request), headers=headers)


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
