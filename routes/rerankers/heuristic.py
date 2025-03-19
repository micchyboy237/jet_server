from fastapi import APIRouter, HTTPException
from jet.file.utils import load_file
from jet.search.similarity import get_bm25_similarities
from jet.search.transformers import clean_string
from typing import List, Dict, Any, TypedDict
from jet.wordnet.n_grams import get_most_common_ngrams
from shared.data_types.job import JobData
from jet.cache.cache_manager import CacheManager
from .reranker_types import SimilarityRequest, SimilarityResult

router = APIRouter()
cache_manager = CacheManager()


class SimilarityRequestData(TypedDict):
    queries: List[str]
    data_file: str


class SimilarityDataItem(TypedDict):
    score: float
    similarity: float
    matched: list[str]
    result: JobData


class SimilarityResultData(TypedDict):
    count: int
    data: List[SimilarityDataItem]


def rerank_bm25(req_queries: list[str], req_data_file: str) -> SimilarityResultData:
    """Processes BM25+ similarity search by handling cache, cleaning data, generating n-grams, and computing similarities."""

    # Load data from file
    data: List[JobData] = load_file(req_data_file)

    # Load previous cache data
    cache_data: Dict[str, Any] = cache_manager.load_cache()

    # Check if cache is valid
    if not cache_manager.is_cache_valid(req_data_file, cache_data):
        sentences: List[str] = [
            clean_string("\n".join([
                item["title"],
                item["details"],
                "\n".join([f"Tech: {tech}" for tech in sorted(
                    item["entities"]["technology_stack"], key=str.lower)]),
                "\n".join([f"Tag: {tech}" for tech in sorted(
                    item["tags"], key=str.lower)]),
            ]).lower())
            for item in data
        ]

        # Generate n-grams
        common_texts_ngrams: List[List[str]] = [
            list(get_most_common_ngrams(sentence, max_words=5).keys()) for sentence in sentences
        ]

        # Update cache
        cache_data = cache_manager.update_cache(
            req_data_file, common_texts_ngrams)
    else:
        common_texts_ngrams: List[List[str]
                                  ] = cache_data["common_texts_ngrams"]

    # Prepare queries
    query_ngrams: List[List[str]] = [
        list(get_most_common_ngrams(query, min_count=1, max_words=5)) for query in req_queries
    ]

    data_dict: Dict[str, JobData] = {item["id"]: item for item in data}
    ids: List[str] = list(data_dict.keys())

    queries: List[str] = ["_".join(text.split())
                          for queries in query_ngrams for text in queries]

    common_texts: List[str] = [" ".join(
        ["_".join(text.split()) for text in texts]) for texts in common_texts_ngrams]

    # Compute BM25+ similarities
    similarities: List[Dict[str, Any]] = get_bm25_similarities(
        queries, common_texts, ids)

    # Format results
    results: List[Dict[str, Any]] = [
        {
            "score": result["score"],
            "similarity": result["similarity"],
            "matched": result["matched"],
            "result": data_dict[result["id"]],
        }
        for result in similarities
    ]

    return {"count": len(results), "data": results}


@router.post("/bm25")
async def bm25_reranker(request: SimilarityRequest) -> SimilarityResult:
    """API endpoint to perform BM25+ similarity ranking."""
    try:
        return rerank_bm25(request.queries, request.data_file)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")
