from fastapi import APIRouter, HTTPException
from jet.file.utils import load_file
from jet.search.similarity import get_bm25_similarities, get_bm25p_similarities
from jet.search.transformers import clean_string
from pydantic import BaseModel
from typing import List
from jet.wordnet.words import get_words
from shared.data_types.job import JobData
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from jet.file.utils import load_file
from jet.vectors.helpers import (
    prepare_sentences,
    setup_colbert_model,
    setup_bert_model,
    setup_cohere_model,
    setup_t5_model,
)
import torch
from .reranker_types import (
    SimilarityRequest,
    SimilarityResult,
)

router = APIRouter()

PHRASE_MODEL_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/wordnet/generated/gensim_jet_phrase_model.pkl"


# **BERT-Based Reranker Similarity Endpoint**
@router.post("/bert", response_model=SimilarityResult)
async def bert_reranker(request: SimilarityRequest):
    try:
        global bert_model
        bert_model = setup_bert_model()  # Ensure the model is initialized

        if bert_model is None:
            raise ValueError("BERT model failed to initialize.")

        data = load_file(request.data_file)

        # Prepare sentences (assumes job descriptions are in data)
        sentences = prepare_sentences(data)

        # Create query-document pairs for scoring
        pairs = [(query, sentence)
                 for query in request.queries for sentence in sentences]

        # Compute similarity scores
        scores = bert_model.predict(pairs)

        # Match scores with sentences
        results = []
        index = 0
        for query in request.queries:
            query_results = []
            for i, sentence in enumerate(sentences):
                score = scores[index]
                index += 1
                query_results.append({
                    "score": score,
                    "similarity": score,
                    "matched": [sentence],
                    "result": data[i]
                })

            # Sort results per query
            query_results.sort(key=lambda x: x["score"], reverse=True)
            results.extend(query_results[:10])  # Keep top 10 matches

        return {"count": len(results), "data": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# **ColBERT Similarity Endpoint**
@router.post("/colbert", response_model=SimilarityResult)
async def colbert_reranker(request: SimilarityRequest):
    try:
        colbert_model = setup_colbert_model()
        data = load_file(request.data_file)
        sentences = prepare_sentences(data)

        # Encode sentences & queries
        sentence_embeddings = colbert_model.encode(
            sentences, convert_to_tensor=True)
        query_embeddings = colbert_model.encode(
            request.queries, convert_to_tensor=True)

        # Compute similarity scores
        similarities = util.cos_sim(query_embeddings, sentence_embeddings)
        top_results = similarities.argsort(descending=True)

        results = [
            {
                "score": similarities[0, idx].item(),
                "similarity": similarities[0, idx].item(),
                "matched": [sentences[idx]],
                "result": data[idx]
            }
            for idx in top_results[0][:10]
        ]

        return {"count": len(results), "data": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# **Cohere Reranker Similarity Endpoint**
@router.post("/cohere", response_model=SimilarityResult)
async def cohere_reranker(request: SimilarityRequest):
    try:
        cohere_model = setup_cohere_model()
        data = load_file(request.data_file)
        sentences = prepare_sentences(data)

        # Call Cohere's reranker API
        response = cohere_model.rerank(
            query=request.queries[0], documents=sentences, top_n=10, model="rerank-english-v2.0")

        results = [
            {
                "score": result.relevance_score,
                "similarity": result.relevance_score,
                "matched": [sentences[result.index]],
                "result": data[result.index]
            }
            for result in response.results
        ]

        return {"count": len(results), "data": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/t5", response_model=SimilarityResult)
async def t5_reranker(request: SimilarityRequest):
    try:
        # Load model and tokenizer
        t5_model, t5_tokenizer = setup_t5_model()

        # Load job data
        data = load_file(request.data_file)
        sentences = prepare_sentences(data)

        results = []

        # Process each query
        for query in request.queries:
            scores = []

            for idx, sentence in enumerate(sentences):
                input_text = f"Query: {query} Document: {sentence} Relevant:"
                inputs = t5_tokenizer(input_text, return_tensors="pt")

                # Generate score
                with torch.no_grad():
                    output = t5_model.generate(**inputs, max_length=2)

                # Convert output to score (0 or 1)
                score = torch.sigmoid(torch.tensor(float(output[0][0]))).item()
                scores.append((score, idx))

            # Sort by highest score
            scores.sort(reverse=True, key=lambda x: x[0])

            # Format results
            query_results = [
                {
                    "score": score,
                    "similarity": score,
                    "matched": [sentences[idx]],
                    "result": data[idx]
                }
                for score, idx in scores[:10]  # Return top 10 results
            ]

            results.extend(query_results)

        return {"count": len(results), "data": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
