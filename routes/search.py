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
    output_dir: str = OUTPUT_DIR


async def stream_progress(event_type: str, message: Optional[str] = None, data: Any = None) -> str:
    """Helper function to format SSE messages with event type."""
    event_data = {}
    if message is not None:
        event_data["message"] = message
    if data is not None:
        event_data["data"] = make_serializable(data)
    sse_message = f"event: {event_type}\n"
    sse_message += f"data: {format_json(event_data)}\n\n"
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

    # Reset searched html results
    if os.path.exists(sub_dir):
        shutil.rmtree(sub_dir)

    yield (await stream_progress("html_processing", "Starting HTML processing", {"total_urls": len(url_html_tuples)}), {})

    for idx, (url, html) in enumerate(url_html_tuples, 1):
        yield (await stream_progress("html_processing", f"Processing HTML {idx}/{len(url_html_tuples)}: {url}"), {})
        output_dir_url = safe_path_from_url(url, sub_dir)
        os.makedirs(output_dir_url, exist_ok=True)

        # Save raw HTML
        save_file(html, os.path.join(output_dir_url, "page.html"))

        header_docs = get_docs_from_html(html)
        save_file("\n\n".join([doc.text for doc in header_docs]), os.path.join(
            output_dir_url, "docs.md"))

        yield (await stream_progress("html_processing", f"Extracted header docs for {url}", {"header_docs_count": len(header_docs)}), {})

        query_scores, reranked_all_nodes = rerank_nodes(
            query, header_docs, embed_models)
        save_file(
            {"url": url, "query": query, "info": compute_info(
                query_scores), "results": query_scores},
            os.path.join(output_dir_url, "query_scores.json")
        )

        yield (
            await stream_progress(
                "html_processing",
                f"Reranked nodes for {url}",
                {"url": url, "query": query, "info": compute_info(
                    query_scores), "results": query_scores}
            ),
            {}
        )

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

        yield (
            await stream_progress(
                "html_processing",
                f"Processed reranked nodes for {url}",
                {"url": url, "query": query, "results": reranked_nodes_data}
            ),
            {}
        )

        html_results.append(
            {"url": url, "query": query, "results": query_scores})
        header_docs_for_all[url] = (
            header_docs, query_scores, reranked_all_nodes)

    yield (await stream_progress("html_processing", "Comparing HTML results"), {})
    comparison_results = compare_html_results(query, html_results)
    save_file(comparison_results, os.path.join(
        output_dir, "comparison_results.json"))

    if not comparison_results:
        yield (await stream_progress("error", "No comparison results available"), {})
        return

    top_result = comparison_results[0]
    top_url = top_result["url"]
    yield (await stream_progress("html_processing", f"Selected top result: {top_url}"), {})

    header_docs, query_scores, reranked_all_nodes = header_docs_for_all[top_url]

    yield (await stream_progress("html_processing", "Grouping nodes for context"), {})
    sorted_reranked_nodes = sorted(
        reranked_all_nodes, key=lambda node: node.metadata['doc_index'])
    grouped_reranked_nodes = group_nodes(sorted_reranked_nodes, "llama3.1")
    context_nodes = grouped_reranked_nodes[0] if grouped_reranked_nodes else []

    yield (
        await stream_progress(
            "html_processing",
            "Context nodes grouped",
            {"context_nodes_count": len(context_nodes)}
        ),
        {}
    )

    # Save context node details
    group_header_doc_indexes = [node.metadata["doc_index"]
                                for node in context_nodes]
    save_file(
        {
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
        },
        os.path.join(output_dir, "reranked_context_nodes.json")
    )

    # Save context markdown
    context = "\n\n".join([node.text for node in context_nodes])
    save_file(context, os.path.join(output_dir, "context.md"))

    final_results = {
        "url": top_url,
        "header_docs": [doc.text for doc in header_docs],
        "html_results": [(url, dir_url, "html_content_omitted") for url, dir_url, _ in html_results],
        "query_scores": query_scores,
        "context_nodes": [{"text": node.text, "score": node.score} for node in context_nodes],
        "reranked_all_nodes": {
            "url": top_url,
            "query": query,
            "info": compute_info(query_scores),
            "results": [
                {
                    "doc": node.metadata["doc_index"] + 1,
                    "rank": rank_idx + 1,
                    "score": node.score,
                    "text": node.text,
                    "metadata": node.metadata,
                }
                for rank_idx, node in enumerate(reranked_all_nodes)
            ]
        }
    }

    yield (
        await stream_progress("html_processing", "Final HTML processing results", final_results),
        final_results
    )


async def process_search(request: SearchRequest) -> AsyncGenerator[str, None]:
    try:
        # Validate inputs
        if not request.query:
            yield await stream_progress("error", "Query cannot be empty")
            return

        if not all(model in OLLAMA_EMBED_MODELS.__args__ for model in request.embed_models):
            yield await stream_progress("error", "Invalid embed model specified")
            return

        # Initialize processing
        os.makedirs(request.output_dir, exist_ok=True)
        yield await stream_progress("progress", "Initialized processing")

        # Perform search and rerank
        yield await stream_progress("search_progress", "Starting search and reranking")
        search_rerank_result = await search_and_filter_data(request.query)
        search_results = search_rerank_result["search_results"]
        url_html_tuples = search_rerank_result["url_html_tuples"]
        save_file(search_results, os.path.join(
            request.output_dir, "search_results.json"))
        yield await stream_progress("search_progress", "Search completed", {"search_results_count": len(search_results)})

        # Process HTMLs
        yield await stream_progress("progress", "Processing HTML content")
        html_generator = process_and_compare_htmls(
            request.query,
            url_html_tuples,
            request.embed_models,
            request.output_dir
        )
        header_docs, html_results, query_scores, context_nodes = [], [], [], []

        async for sse_message, data in html_generator:
            yield sse_message
            if data and "header_docs" in data:
                header_docs = [BaseDocument(text=text)
                               for text in data["header_docs"]]
                html_results = [(url, dir_url, html)
                                for url, dir_url, html in data["html_results"]]
                query_scores = data["query_scores"]
                context_nodes = [NodeWithScore(node=TextNode(text=node["text"]), score=node["score"])
                                 for node in data["context_nodes"]]
                reranked_all_nodes = data["reranked_all_nodes"]

        # Validate outputs
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

        yield await stream_progress(
            "progress",
            "HTML processing completed",
            {"header_docs_count": len(header_docs),
             "html_results_count": len(html_results)}
        )

        # Save top results
        save_file(
            make_serializable("\n\n".join([doc.text for doc in header_docs])),
            os.path.join(request.output_dir, "top_docs.md")
        )
        save_file(
            make_serializable({"url": data["url"], "query": request.query, "info": compute_info(
                query_scores), "results": query_scores}),
            os.path.join(request.output_dir, "top_query_scores.json")
        )
        save_file(
            {"url": data["url"], "query": request.query,
                "results": reranked_all_nodes},
            os.path.join(request.output_dir, "top_reranked_nodes.json")
        )

        # Prepare header texts
        yield await stream_progress("progress", "Extracting header content")
        header_texts = [doc.text for doc in header_docs]
        headers_text = "\n\n".join(header_texts)
        yield await stream_progress("progress", "Header content extracted", {"header_text_length": len(headers_text)})

        # Send query scores
        yield await stream_progress("progress", "Sending query scores", {"query": request.query, "results": query_scores})

        # Prepare context node details
        yield await stream_progress("progress", "Processing context nodes")
        group_header_doc_indexes = [node.node.metadata.get(
            "doc_index", i) for i, node in enumerate(context_nodes)]
        context_nodes_data = [
            {
                "doc": node.node.metadata.get("doc_index", i) + 1,
                "rank": rank_idx + 1,
                "score": node.score,
                "text": node.node.text,
                "metadata": node.node.metadata,
            }
            for rank_idx, node in enumerate(context_nodes)
            if node.node.metadata.get("doc_index", rank_idx) in group_header_doc_indexes
        ]
        yield await stream_progress("progress", "Context nodes processed", {"query": request.query, "results": context_nodes_data})

        # Prepare context markdown
        yield await stream_progress("progress", "Generating context markdown")
        context = "\n\n".join([node.node.text for node in context_nodes])
        yield await stream_progress("progress", "Context markdown generated", {"context_length": len(context)})

        # Signal start of LLM streaming
        yield await stream_progress("chat_start", "Starting LLM streaming response")

        # Stream LLM response
        llm = Ollama(temperature=0.3, model=request.llm_model)
        async for chunk in llm.stream_chat(
            query=request.query,
            context=context,
            model=request.llm_model,
        ):
            yield await stream_progress("chat_chunk", None, chunk)

        # Signal end of LLM streaming
        yield await stream_progress("chat_end", "LLM streaming response completed")

        # Final completion
        yield await stream_progress(
            "completed",
            "Processing completed",
            {
                "status": "success",
                "query": request.query,
                "header_docs_count": len(header_docs),
                "context_nodes_count": len(context_nodes)
            }
        )

    except Exception as e:
        yield await stream_progress("error", f"Error processing request: {str(e)}")
        return


@router.post("/search-and-process")
async def search_and_process(request: SearchRequest):
    return StreamingResponse(
        process_search(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
