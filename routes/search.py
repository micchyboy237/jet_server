import asyncio
import traceback
from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse
from jet.features.eval_search_and_chat import evaluate_llm_response
from jet.features.eval_tasks import enqueue_evaluation_task
from jet.transformers.object import make_serializable
from jet.transformers.formatters import format_json
from jet.utils.collection_utils import group_by
from jet.wordnet.similarity import compute_info
from pydantic import BaseModel
from typing import List, Dict, Any, AsyncGenerator, Literal, Optional, Tuple
import os
import json
import shutil
from llama_index.core.schema import Document as BaseDocument, NodeWithScore
from jet.features.search_and_chat import compare_html_query_scores, search_and_filter_data, truncate_docs
from jet.llm.models import OLLAMA_EMBED_MODELS, OLLAMA_MODEL_NAMES
from jet.scrapers.utils import safe_path_from_url
from jet.llm.ollama.base import Ollama
from jet.features.search_and_chat import compare_html_results, get_docs_from_html, rerank_nodes, group_nodes
from llama_index.core.schema import TextNode
from jet.file.utils import save_file
from jet.llm.evaluators.context_relevancy_evaluator import evaluate_context_relevancy
from jet.llm.evaluators.answer_relevancy_evaluator import evaluate_answer_relevancy

router = APIRouter()

OUTPUT_DIR = "generated/search"


class SearchRequest(BaseModel):
    query: str
    embed_models: List[OLLAMA_EMBED_MODELS] = [
        "all-minilm:33m", "paraphrase-multilingual"]
    llm_model: OLLAMA_MODEL_NAMES = "llama3.2"
    format: Optional[str | Literal["json"]] = None


async def stream_progress(event_type: str, description: Optional[str] = None, data: Any = None) -> str:
    """Helper function to format SSE messages with event type."""
    sse_message = f"event: {event_type}\n"
    payload = data if data is not None else description
    sse_message += f"data: {format_json(make_serializable(payload), indent=None)}\n\n"
    return sse_message


async def process_search(request: SearchRequest, session_id: Optional[str] = None) -> AsyncGenerator[str, None]:

    try:
        query = request.query
        embed_models = request.embed_models
        llm_model = request.llm_model
        format = request.format
        output_dir = os.path.join(OUTPUT_DIR, query.lower().replace(' ', '_'))

        if not query:
            yield await stream_progress("validation_error", "Query cannot be empty")
            return

        if not all(model in OLLAMA_EMBED_MODELS.__args__ for model in embed_models):
            yield await stream_progress("validation_error", "Invalid embed model specified")
            return

        if not session_id:
            os.makedirs(output_dir, exist_ok=True)
            yield await stream_progress("start", "Initialized processing")
            yield await stream_progress("start", "Starting search and reranking")

            search_rerank_result = search_and_filter_data(query)
            search_results = search_rerank_result["search_results"]
            url_html_tuples = search_rerank_result["url_html_tuples"]
            save_file(search_results, os.path.join(
                output_dir, "search_results.json"))

            yield await stream_progress("search_results", "Search completed", {"search_results_count": len(search_results)})

            yield await stream_progress("comparison_start", "Comparing HTML results")

            comparison_results = compare_html_query_scores(
                query, url_html_tuples, embed_models)

            top_urls = comparison_results["top_urls"]
            top_query_scores = comparison_results["top_query_scores"]

            yield await stream_progress("comparison_complete", f"Selected top result: {top_urls}", top_query_scores)

            top_reranked_nodes: list[NodeWithScore] = []
            for item in top_query_scores:
                top_reranked_nodes.append(NodeWithScore(
                    node=TextNode(
                        node_id=item["id"],
                        text=item["text"],
                        metadata=item["metadata"]
                    ),
                    score=item["score"]
                ))

            grouped_reranked_nodes = group_nodes(top_reranked_nodes, llm_model)
            top_context_nodes = grouped_reranked_nodes[0] if grouped_reranked_nodes else [
            ]
            top_grouped_context_nodes = group_by(
                top_context_nodes, "metadata['url']")
            sorted_context_nodes: list[NodeWithScore] = []
            sorted_contexts: list[str] = []
            for grouped_nodes in top_grouped_context_nodes:
                nodes_with_scores: List[NodeWithScore] = grouped_nodes["items"]
                sorted_nodes_with_scores = sorted(
                    nodes_with_scores, key=lambda node: node.metadata['doc_index'])
                sorted_context_nodes.extend(sorted_nodes_with_scores)
                sorted_contexts.extend(
                    [node.text for node in sorted_nodes_with_scores])

            save_file({
                "url": top_urls,
                "query": query,
                "info": compute_info(top_query_scores),
                "results": top_query_scores
            }, os.path.join(output_dir, "top_query_scores.json"))

            save_file({
                "url": top_urls,
                "query": query,
                "results": [
                    {
                        "doc_index": node.metadata["doc_index"],
                        "node_id": node.node_id,
                        "url": node.metadata["url"],
                        "score": node.score,
                        "text": node.text,
                    }
                    for node in sorted_context_nodes
                ]
            }, os.path.join(output_dir, "top_context_nodes.json"))

            context = "\n\n".join(sorted_contexts)
            save_file(context, os.path.join(output_dir, "top_context.md"))
        else:
            context = None

        yield await stream_progress("start", "Starting LLM streaming response")
        llm = Ollama(temperature=0.3, model=llm_model, session_id=session_id)
        response = ""
        async for chunk in llm.stream_chat(query=query, context=context, model=llm_model, format=format):
            response += chunk
            yield await stream_progress("chunk", None, chunk)
        save_file(response, os.path.join(output_dir, "chat_response.md"))

        yield await stream_progress("chat_complete", "LLM streaming response completed")

        save_file({"query": query, "context": context, "response": response},
                  os.path.join(output_dir, "summary.json"))

        # enqueue_evaluation_task(query, response, context, embed_model=embed_models[0],
        #                         llm_model=llm_model, output_dir=output_dir)

        yield await stream_progress("complete", "Processing completed", {
            "status": "success",
            "query": query,
            "context_nodes_count": len(top_context_nodes),
            "session_id": llm.session_id
        })

    except Exception as e:
        yield await stream_progress("error", f"Error processing request: {str(e)}")
        traceback.print_exc()
        return


@router.post("/search-and-process")
async def search_and_process(
    request: SearchRequest,
    session_id: Optional[str] = Header(default=None, alias="session-id")
):
    return StreamingResponse(
        process_search(request, session_id=session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
