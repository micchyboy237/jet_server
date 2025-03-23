from fastapi import APIRouter, HTTPException
from jet.file.utils import load_file
from jet.search.similarity import get_bm25_similarities
from jet.search.transformers import clean_string
from typing import List, Dict, Any, Optional, TypedDict
from jet.utils.object import extract_values_by_paths
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


def format_texts(object: dict[str, Any], json_attributes: list[str] = [], exclude_json_attributes: list[str] = []):
    if json_attributes:
        json_parts_dict = extract_values_by_paths(
            object, json_attributes, is_flattened=True) if json_attributes else None
        text_parts = []
        for key, value in json_parts_dict.items():
            value_str = str(value)
            if isinstance(value, list):
                value_str = ", ".join(value)
            if value_str.strip():
                text_parts.append(
                    f"{key.title().replace('_', ' ')}: {value_str}")
    else:
        # Use all attributes
        text_parts = [f"{key.title().replace('_', ' ')}: {str(value)}"
                      for key, value in object.items()
                      if key not in exclude_json_attributes and value]

    text_content = "\n".join(text_parts) if text_parts else ""
    return text_content


@router.post("/bm25")
async def bm25_reranker(request: SimilarityRequest) -> SimilarityResult:
    """API endpoint to perform BM25+ similarity ranking."""
    data: list[str | dict[str, Any]] = load_file(request.data_file)
    texts = [format_texts(obj) if isinstance(
        obj, dict) else obj for obj in data]

    similarity_results = get_bm25_similarities(request.queries, texts)
    return {
        "count": len(similarity_results),
        "data": similarity_results
    }
