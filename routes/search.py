from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, AsyncGenerator, Tuple
import os
import json
from llama_index.core.schema import Document as BaseDocument, NodeWithScore
from jet.features.search_and_chat import search_and_rerank_data
from jet.llm.models import OLLAMA_EMBED_MODELS
from jet.scrapers.utils import safe_path_from_url
from jet.llm.ollama.base import Ollama
# Note: These imports are assumed; replace with actual implementations
from jet.features.search_and_chat import compare_html_results, get_docs_from_html, rerank_nodes, group_nodes

router = APIRouter()

OUTPUT_DIR = "generated"


class SearchRequest(BaseModel):
    query: str
    embed_models: List[str] = ["mxbai-embed-large", "paraphrase-multilingual"]
    llm_model: str = "llama3.1"
    output_dir: str = OUTPUT_DIR


async def stream_progress(message: str, data: Any = None) -> str:
    """Helper function to format SSE messages."""
    event_data = {"message": message}
    if data is not None:
        event_data["data"] = data
    return f"data: {json.dumps(event_data)}\n\n"


async def process_and_compare_htmls(
    query: str,
    selected_html: List[Tuple[str, str]],
    embed_models: List[OLLAMA_EMBED_MODELS],
    output_dir: str
) -> AsyncGenerator[Tuple[str, Dict[str, Any]], None]:
    """
    Process HTMLs, rerank documents, and compare results to find the best-matching HTML, yielding progress updates.

    Yields:
        Tuple[str, Dict[str, Any]]: SSE-formatted progress message and optional data, including final results.
    """
    html_results = []
    header_docs_for_all = {}
    sub_dir = os.path.join(output_dir, "searched_html")

    yield (await stream_progress("Starting HTML processing", {"total_urls": len(selected_html)}), {})

    for idx, (url, html) in enumerate(selected_html, 1):
        yield (await stream_progress(f"Processing HTML {idx}/{len(selected_html)}: {url}"), {})
        output_dir_url = safe_path_from_url(url, sub_dir)
        os.makedirs(output_dir_url, exist_ok=True)

        header_docs = get_docs_from_html(html)
        yield (await stream_progress(f"Extracted header docs for {url}", {"header_docs_count": len(header_docs)}), {})

        query_scores, reranked_all_nodes = rerank_nodes(
            query, header_docs, embed_models)
        yield (
            await stream_progress(
                f"Reranked nodes for {url}",
                {"query": query, "results": query_scores}
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
        yield (
            await stream_progress(
                f"Processed reranked nodes for {url}",
                {"query": query, "results": reranked_nodes_data}
            ),
            {}
        )

        html_results.append((url, output_dir_url, html))
        header_docs_for_all[url] = (
            header_docs, query_scores, reranked_all_nodes)

    yield (await stream_progress("Comparing HTML results"), {})
    comparison_results = compare_html_results(query, html_results, top_n=1)

    if not comparison_results:
        yield (await stream_progress("Error: No comparison results available"), {})
        return

    top_result = comparison_results[0]
    top_url = top_result["url"]
    yield (await stream_progress(f"Selected top result: {top_url}"), {})

    header_docs, query_scores, reranked_all_nodes = header_docs_for_all[top_url]

    yield (await stream_progress("Grouping nodes for context"), {})
    sorted_reranked_nodes = sorted(
        reranked_all_nodes, key=lambda node: node.metadata['doc_index'])
    grouped_reranked_nodes = group_nodes(sorted_reranked_nodes, "llama3.1")
    context_nodes = grouped_reranked_nodes[0] if grouped_reranked_nodes else []
    yield (
        await stream_progress(
            "Context nodes grouped",
            {"context_nodes_count": len(context_nodes)}
        ),
        {}
    )

    # Yield final results as a special message
    final_results = {
        # Simplified for brevity
        "header_docs": [doc.text for doc in header_docs],
        "html_results": [(url, dir_url, "html_content_omitted") for url, dir_url, _ in html_results],
        "query_scores": query_scores,
        "context_nodes": [{"text": node.text, "score": node.score} for node in context_nodes]
    }
    yield (
        await stream_progress("Final HTML processing results", final_results),
        final_results
    )


async def process_search(request: SearchRequest) -> AsyncGenerator[str, None]:
    try:
        # Validate inputs
        if not request.query:
            yield await stream_progress("Error: Query cannot be empty")
            return

        if not all(model in OLLAMA_EMBED_MODELS.__args__ for model in request.embed_models):
            yield await stream_progress("Error: Invalid embed model specified")
            return

        # Create output directory if needed
        os.makedirs(request.output_dir, exist_ok=True)
        yield await stream_progress("Initialized processing")

        # Perform search and rerank
        yield await stream_progress("Starting search and reranking")
        search_results, selected_html = search_and_rerank_data(request.query)
        yield await stream_progress("Search completed", {"search_results_count": len(search_results)})

        # Process HTMLs and get results
        yield await stream_progress("Processing HTML content")
        html_generator = process_and_compare_htmls(
            request.query,
            selected_html,
            request.embed_models,
            request.output_dir
        )
        header_docs, html_results, query_scores, context_nodes = [], [], [], []

        async for sse_message, data in html_generator:
            yield sse_message
            if data and "header_docs" in data:
                # Extract final results from the special message
                header_docs = [BaseDocument(text=text)
                               for text in data["header_docs"]]
                html_results = [(url, dir_url, html)
                                for url, dir_url, html in data["html_results"]]
                query_scores = data["query_scores"]
                context_nodes = [NodeWithScore(node=TextNode(
                    text=node["text"]), score=node["score"]) for node in data["context_nodes"]]

        yield await stream_progress(
            "HTML processing completed",
            {"header_docs_count": len(header_docs),
             "html_results_count": len(html_results)}
        )

        # Prepare header texts
        yield await stream_progress("Extracting header content")
        header_texts = [doc.text for doc in header_docs]
        headers_text = "\n\n".join(header_texts)
        yield await stream_progress("Header content extracted", {"header_text_length": len(headers_text)})

        # Send query scores
        yield await stream_progress("Sending query scores", {"query": request.query, "results": query_scores})

        # Prepare context node details
        yield await stream_progress("Processing context nodes")
        group_header_doc_indexes = [node.metadata.get(
            "doc_index", i) for i, node in enumerate(context_nodes)]
        context_nodes_data = [
            {
                "doc": node.metadata.get("doc_index", i) + 1,
                "rank": rank_idx + 1,
                "score": node.score,
                "text": node.text,
                "metadata": node.metadata,
            }
            for rank_idx, node in enumerate(context_nodes)
            if node.metadata.get("doc_index", rank_idx) in group_header_doc_indexes
        ]
        yield await stream_progress("Context nodes processed", {"query": request.query, "results": context_nodes_data})

        # Prepare context markdown
        yield await stream_progress("Generating context markdown")
        context = "\n\n".join([node.text for node in context_nodes])
        yield await stream_progress("Context markdown generated", {"context_length": len(context)})

        # Run LLM response
        yield await stream_progress("Generating LLM response")
        llm = Ollama(temperature=0.3, model=request.llm_model)
        response = llm.chat(
            request.query,
            context=context,
            model=request.llm_model,
        )
        yield await stream_progress("LLM response generated", {"query": request.query, "response": response})

        # Final completion message
        yield await stream_progress(
            "Processing completed",
            {
                "status": "success",
                "query": request.query,
                "header_docs_count": len(header_docs),
                "context_nodes_count": len(context_nodes)
            }
        )

    except Exception as e:
        yield await stream_progress(f"Error processing request: {str(e)}")
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
