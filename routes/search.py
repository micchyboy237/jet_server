from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from jet.transformers.object import make_serializable
from jet.transformers.formatters import format_json
from jet.wordnet.similarity import compute_info
from pydantic import BaseModel
from typing import List, Dict, Any, AsyncGenerator, Optional, Tuple
import os
import json
import shutil
from llama_index.core.schema import Document as BaseDocument, NodeWithScore
from jet.features.search_and_chat import search_and_filter_data
from jet.llm.models import OLLAMA_EMBED_MODELS
from jet.scrapers.utils import safe_path_from_url
from jet.llm.ollama.base import Ollama
from jet.features.search_and_chat import compare_html_results, get_docs_from_html, rerank_nodes, group_nodes
from llama_index.core.schema import TextNode
from jet.file.utils import save_file

router = APIRouter()

OUTPUT_DIR = "generated/search"


class SearchRequest(BaseModel):
    query: str
    embed_models: List[str] = ["mxbai-embed-large", "paraphrase-multilingual"]
    llm_model: str = "llama3.1"


async def stream_progress(event_type: str, description: Optional[str] = None, data: Any = None) -> str:
    """Helper function to format SSE messages with event type."""
    sse_message = f"event: {event_type}\n"
    payload = data if data is not None else description
    sse_message += f"data: {format_json(make_serializable(payload))}\n\n"
    return sse_message


async def process_and_compare_htmls(
    query: str,
    url_html_tuples: List[Tuple[str, str]],
    embed_models: List[OLLAMA_EMBED_MODELS],
    output_dir: str
) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
    html_results = []
    header_docs_for_all = {}
    sub_dir = os.path.join(output_dir, "searched_html")

    if os.path.exists(sub_dir):
        shutil.rmtree(sub_dir)

    yield (await stream_progress("start", "Starting HTML processing", {"total_urls": len(url_html_tuples)}), {})

    for idx, (url, html) in enumerate(url_html_tuples, 1):
        yield (await stream_progress("url_start", f"Processing HTML {idx}/{len(url_html_tuples)}: {url}"), {})

        output_dir_url = safe_path_from_url(url, sub_dir)
        os.makedirs(output_dir_url, exist_ok=True)

        save_file(html, os.path.join(output_dir_url, "page.html"))

        header_docs = get_docs_from_html(html)
        save_file("\n\n".join([doc.text for doc in header_docs]), os.path.join(
            output_dir_url, "docs.md"))

        yield (await stream_progress("docs_extracted", f"Extracted header docs for {url}", {"header_docs_count": len(header_docs)}), {})

        query_scores, reranked_all_nodes = rerank_nodes(
            query, header_docs, embed_models)
        save_file(
            {"url": url, "query": query, "info": compute_info(
                query_scores), "results": query_scores},
            os.path.join(output_dir_url, "query_scores.json")
        )

        yield (await stream_progress("nodes_reranked", f"Reranked nodes for {url}", query_scores), {})

        reranked_nodes_data = [
            {
                "doc": node.metadata["doc_index"] + 1,
                "rank": rank_idx + 1,
                "score": node.score,
                "text": node.text,
                "metadata": node.metadata,
            }
            for rank_idx, node in enumerate(reranked_all_nodes)
        ]
        save_file(
            {"url": url, "query": query, "results": reranked_nodes_data},
            os.path.join(output_dir_url, "reranked_all_nodes.json")
        )

        yield (await stream_progress("nodes_processed", f"Processed reranked nodes for {url}", reranked_nodes_data), {})

        html_results.append(
            {"url": url, "query": query, "results": query_scores})
        header_docs_for_all[url] = (
            header_docs, query_scores, reranked_all_nodes)

    yield (await stream_progress("comparison_start", "Comparing HTML results"), {})
    comparison_results = compare_html_results(query, html_results)
    save_file(comparison_results, os.path.join(
        output_dir, "comparison_results.json"))

    if not comparison_results:
        yield (await stream_progress("error", "No comparison results available"), {})
        return

    top_result = comparison_results[0]
    top_url = top_result["url"]
    yield (await stream_progress("comparison_complete", f"Selected top result: {top_url}"), {})

    header_docs, query_scores, reranked_all_nodes = header_docs_for_all[top_url]

    yield (await stream_progress("nodes_grouping", "Grouping nodes for context"), {})

    sorted_reranked_nodes = sorted(
        reranked_all_nodes, key=lambda node: node.metadata['doc_index'])
    grouped_reranked_nodes = group_nodes(sorted_reranked_nodes, "llama3.1")
    context_nodes = grouped_reranked_nodes[0] if grouped_reranked_nodes else []

    yield (await stream_progress("nodes_grouped", "Context nodes grouped", {"context_nodes_count": len(context_nodes)}), {})

    group_header_doc_indexes = [node.metadata["doc_index"]
                                for node in context_nodes]
    save_file({
        "query": query,
        "results": [
            {
                "doc": node.metadata["doc_index"] + 1,
                "rank": rank_idx + 1,
                "score": node.score,
                "text": node.text,
                "metadata": node.metadata,
            }
            for rank_idx, node in enumerate(context_nodes)
            if node.metadata["doc_index"] in group_header_doc_indexes
        ]
    }, os.path.join(output_dir, "reranked_context_nodes.json"))

    context = "\n\n".join([node.text for node in context_nodes])
    save_file(context, os.path.join(output_dir, "context.md"))

    final_results = {
        "url": top_url,
        "header_docs": [doc.text for doc in header_docs],
        "query_scores": query_scores,
        "context_nodes": context_nodes,
        "reranked_all_nodes": reranked_all_nodes
    }

    yield (await stream_progress("complete", "Final HTML processing results", final_results), final_results)


async def process_search(request: SearchRequest) -> AsyncGenerator[str, None]:
    try:
        query = request.query
        output_dir = os.path.join(OUTPUT_DIR, query.lower().replace(' ', '_'))

        if not query:
            yield await stream_progress("validation_error", "Query cannot be empty")
            return

        if not all(model in OLLAMA_EMBED_MODELS.__args__ for model in request.embed_models):
            yield await stream_progress("validation_error", "Invalid embed model specified")
            return

        os.makedirs(output_dir, exist_ok=True)
        yield await stream_progress("start", "Initialized processing")
        yield await stream_progress("start", "Starting search and reranking")

        search_rerank_result = await search_and_filter_data(query)
        search_results = search_rerank_result["search_results"]
        url_html_tuples = search_rerank_result["url_html_tuples"]
        save_file(search_results, os.path.join(
            output_dir, "search_results.json"))

        yield await stream_progress("results", "Search completed", {"search_results_count": len(search_results)})
        yield await stream_progress("start", "Processing HTML content")

        html_generator = process_and_compare_htmls(
            query, url_html_tuples, request.embed_models, output_dir)
        header_docs, html_results, query_scores, context_nodes = [], [], [], []

        async for sse_message, data in html_generator:
            yield sse_message
            if data and "header_docs" in data:
                url = data["url"]
                header_docs = [BaseDocument(text=text)
                               for text in data["header_docs"]]
                query_scores = data["query_scores"]
                context_nodes = data["context_nodes"]
                reranked_all_nodes = data["reranked_all_nodes"]

        missing_parts = []
        if not header_docs:
            missing_parts.append("header_docs")
        if not query_scores:
            missing_parts.append("query_scores")
        if not context_nodes:
            missing_parts.append("context_nodes")

        if missing_parts:
            yield await stream_progress("error", f"Missing data in: {', '.join(missing_parts)}")
            return

        yield await stream_progress("complete", "HTML processing completed", {
            "header_docs_count": len(header_docs),
            "html_results_count": len(html_results)
        })

        save_file("\n\n".join([doc.text for doc in header_docs]), os.path.join(
            output_dir, "top_docs.md"))
        save_file(make_serializable({
            "url": url,
            "query": query,
            "info": compute_info(query_scores),
            "results": query_scores
        }), os.path.join(output_dir, "top_query_scores.json"))

        save_file({"url": url, "query": query, "results": reranked_all_nodes},
                  os.path.join(output_dir, "top_reranked_nodes.json"))

        yield await stream_progress("start", "Extracting header content")
        header_texts = [doc.text for doc in header_docs]
        headers_text = "\n\n".join(header_texts)
        yield await stream_progress("complete", "Header content extracted", {"header_text_length": len(headers_text)})

        yield await stream_progress("scores", "Sending query scores", {"query": query, "results": query_scores})
        yield await stream_progress("start", "Generating context markdown")
        context = "\n\n".join([node.text for node in context_nodes])
        yield await stream_progress("complete", "Context markdown generated", {"context_length": len(context)})

        yield await stream_progress("start", "Starting LLM streaming response")
        llm = Ollama(temperature=0.3, model=request.llm_model)
        response = ""
        async for chunk in llm.stream_chat(query=query, context=context, model=request.llm_model):
            response += chunk
            yield await stream_progress("chunk", None, chunk)
        save_file(response, os.path.join(output_dir, "chat_response.md"))

        yield await stream_progress("complete", "LLM streaming response completed")
        yield await stream_progress("complete", "Processing completed", {
            "status": "success",
            "query": query,
            "header_docs_count": len(header_docs),
            "context_nodes_count": len(context_nodes)
        })

    except Exception as e:
        yield await stream_progress("error", f"Error processing request: {str(e)}")
        return


@router.post("/search-and-process")
async def search_and_process(request: SearchRequest):
    return StreamingResponse(
        process_search(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
