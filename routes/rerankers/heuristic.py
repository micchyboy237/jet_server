from fastapi import APIRouter, HTTPException
from jet.file.utils import load_file
from jet.search.similarity import get_bm25_similarities
from jet.search.transformers import clean_string
from typing import List
from jet.wordnet.n_grams import get_most_common_ngrams
from jet.wordnet.words import get_words
from shared.data_types.job import JobData
from .reranker_types import SimilarityRequest, SimilarityResult
from .cache_manager import CacheManager

router = APIRouter()

cache_manager = CacheManager()


# BM25+ Similarity search endpoint (POST method)
@router.post("/bm25", response_model=SimilarityResult)
async def bm25_reranker(request: SimilarityRequest):
    try:
        # Load data
        data: List[JobData] = load_file(request.data_file)

        # Load previous cache data
        cache_data = cache_manager.load_cache()

        # Check if the cache is valid or needs to be updated
        if not cache_manager.is_cache_valid(request.data_file, cache_data):
            # Cache is not valid, regenerate n-grams
            sentences = []
            for item in data:
                sentence = "\n".join([
                    item["title"],
                    item["details"],
                    "\n".join([f"Tech: {tech}" for tech in sorted(
                        item["entities"]["technology_stack"], key=str.lower)]),
                    "\n".join([f"Tag: {tech}" for tech in sorted(
                        item["tags"], key=str.lower)]),
                ])
                cleaned_sentence = clean_string(sentence.lower())
                sentences.append(cleaned_sentence)

            # Generate n-grams
            common_texts_ngrams = [
                list(get_most_common_ngrams(sentence, max_words=5).keys()) for sentence in sentences
            ]

            # Update the cache with the new n-grams
            cache_data = cache_manager.update_cache(
                request.data_file, common_texts_ngrams)
        else:
            # Use the cached n-grams
            common_texts_ngrams = cache_data["common_texts_ngrams"]

        # Prepare queries and calculate BM25+ similarities
        query_ngrams = [list(get_most_common_ngrams(
            query, min_count=1, max_words=5)) for query in request.queries]
        data_dict = {item["id"]: item for item in data}
        ids = list(data_dict.keys())
        queries = ["_".join(text.split())
                   for queries in query_ngrams for text in queries]

        common_texts = []
        for texts in common_texts_ngrams:
            formatted_texts = []
            for text in texts:
                formatted_texts.append("_".join(text.split()))
            common_texts.append(" ".join(formatted_texts))

        similarities = get_bm25_similarities(queries, common_texts, ids)

        # Format the results
        results = [
            {
                "score": result["score"],
                "similarity": result["similarity"],
                "matched": result["matched"],
                "result": data_dict[result["id"]]
            }
            for result in similarities
        ]

        return {
            "count": len(results),
            "data": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}"
        )
