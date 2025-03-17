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
    setup_phrase_detector,
    prepare_sentences,
    setup_colbert_model,
    setup_bert_model,
    setup_cohere_model,
    setup_t5_model,
)
import torch

router = APIRouter()

PHRASE_MODEL_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/wordnet/generated/gensim_jet_phrase_model.pkl"


# Define Request Model
class SimilarityRequest(BaseModel):
    queries: List[str]
    data_file: str = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"


# Define Response Models
class SimilarityData(BaseModel):
    score: float
    similarity: float
    matched: list[str]
    result: JobData


class SimilarityResult(BaseModel):
    count: int
    data: List[SimilarityData]


# BM25 Similarity search endpoint (POST method)
@router.post("/bm25-reranker", response_model=SimilarityResult)
async def bm25_reranker(request: SimilarityRequest):
    try:
        # Load data
        data: List[JobData] = load_file(request.data_file)

        # Prepare sentences
        sentences = []
        sentences_dict = {}
        sentences_no_newline = []

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
            sentences_no_newline.append(" ".join(get_words(cleaned_sentence)))

        # Perform BM25 similarity search
        queries = ["_".join(get_words(query.lower()))
                   for query in request.queries]

        detector = setup_phrase_detector(sentences)

        results_generator = detector.detect_phrases(sentences)
        for result in results_generator:
            multi_gram_phrases = " ".join(result["phrases"])
            orig_sentence = sentences_no_newline[result["index"]]
            updated_sentence = orig_sentence + " " + multi_gram_phrases

            orig_data = data[result["index"]]
            sentences_dict[updated_sentence] = orig_data
            sentences_no_newline[result["index"]] = updated_sentence

        similarities = get_bm25_similarities(
            queries, sentences_no_newline)

        # Format the results
        results = [
            {
                "score": result["score"],
                "similarity": result["similarity"],
                "matched": result["matched"],
                "result": sentences_dict[result["text"]]
            }
            for result in similarities
        ]

        return {
            "count": len(results),
            "data": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")


# BM25+ Similarity search endpoint (POST method)
# Updated endpoint name
@router.post("/bm25p-reranker", response_model=SimilarityResult)
async def bm25p_reranker(request: SimilarityRequest):
    try:
        # Load data
        data: List[JobData] = load_file(request.data_file)

        # Prepare sentences
        sentences = []
        sentences_dict = {}
        sentences_no_newline = []

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
            sentences_no_newline.append(" ".join(get_words(cleaned_sentence)))

        # Perform BM25+ similarity search
        queries = ["_".join(get_words(query.lower()))
                   for query in request.queries]

        detector = setup_phrase_detector(sentences)

        results_generator = detector.detect_phrases(sentences)
        for result in results_generator:
            multi_gram_phrases = " ".join(result["phrases"])
            orig_sentence = sentences_no_newline[result["index"]]
            updated_sentence = orig_sentence + " " + multi_gram_phrases

            orig_data = data[result["index"]]
            sentences_dict[updated_sentence] = orig_data
            sentences_no_newline[result["index"]] = updated_sentence

        similarities = get_bm25p_similarities(  # Updated function call
            queries, sentences_no_newline
        )

        # Format the results
        results = [
            {
                "score": result["score"],
                "similarity": result["similarity"],
                "matched": result["matched"],
                "result": sentences_dict[result["text"]]
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


# **BERT-Based Reranker Similarity Endpoint**
@router.post("/bert-reranker", response_model=SimilarityResult)
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
@router.post("/colbert-reranker", response_model=SimilarityResult)
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
@router.post("/cohere-reranker", response_model=SimilarityResult)
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


@router.post("/t5-reranker", response_model=SimilarityResult)
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
