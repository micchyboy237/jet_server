import hashlib
import json
import os
import pickle
from tqdm import tqdm
from fastapi import APIRouter, HTTPException
from jet.file.utils import load_file
from jet.search.similarity import get_bm25p_similarities
from jet.search.transformers import clean_string
from typing import List
from jet.wordnet.n_grams import get_most_common_ngrams
from jet.wordnet.words import get_words
from shared.data_types.job import JobData
from .reranker_types import (
    SimilarityRequest,
    SimilarityResult,
)

router = APIRouter()

# Set the directory for cached files
CACHE_DIR = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_server/.cache/heuristics"
CACHE_FILE = "ngrams_cache.pkl"  # Name of the cache file


def get_file_hash(file_path: str) -> str:
    """Generate a hash for the given file."""
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_cache() -> dict:
    """Load the cache file if exists, otherwise return an empty dict."""
    cache_path = os.path.join(CACHE_DIR, CACHE_FILE)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as cache_file:
            return pickle.load(cache_file)
    return {}


def save_cache(data: dict) -> None:
    """Save the cache to a file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, CACHE_FILE)
    with open(cache_path, 'wb') as cache_file:
        pickle.dump(data, cache_file)

# BM25+ Similarity search endpoint (POST method)


@router.post("/bm25", response_model=SimilarityResult)
async def bm25_reranker(request: SimilarityRequest):
    try:
        # Load data
        data: List[JobData] = load_file(request.data_file)

        # Generate hash for the current data file
        current_file_hash = get_file_hash(request.data_file)

        # Load previous cache data
        cache_data = load_cache()

        # Check if the data file has been updated
        if cache_data.get("file_hash") != current_file_hash:
            # File has been updated, regenerate the n-grams
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
                list(get_most_common_ngrams(sentence, max_words=5).keys()) for sentence in tqdm(sentences)
            ]

            # Save the new n-grams and file hash in the cache
            cache_data = {
                "file_hash": current_file_hash,
                "common_texts_ngrams": common_texts_ngrams
            }
            save_cache(cache_data)
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

        similarities = get_bm25p_similarities(queries, common_texts, ids)

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
