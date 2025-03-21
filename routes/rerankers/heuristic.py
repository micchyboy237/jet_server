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


@router.post("/bm25")
async def bm25_reranker(request: SimilarityRequest) -> SimilarityResult:
    """API endpoint to perform BM25+ similarity ranking."""
    try:
        return rerank_bm25(request.queries, request.data_file)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")
