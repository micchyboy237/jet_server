from fastapi import APIRouter, HTTPException
from jet.file.utils import load_file
from jet.search.similarity import get_bm25_similarities
from jet.search.transformers import clean_string
from pydantic import BaseModel
from typing import List, TypedDict
from jet.wordnet.words import get_words
from shared.data_types.job import JobData

router = APIRouter()

PHRASE_MODEL_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/wordnet/generated/gensim_jet_phrase_model.pkl"

phrase_detector = None

# Define the request body model


class BM25SimilarityRequest(BaseModel):
    queries: List[str]
    data_file: str = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"


# Define the response model
class BM25SimilarityData(BaseModel):
    score: float
    similarity: float
    matched: list[str]
    text: str
    result: JobData


class BM25SimilarityResult(BaseModel):
    count: int
    data: List[BM25SimilarityData]


def setup_phrase_detector(sentences):
    from jet.wordnet.gensim_scripts.phrase_detector import PhraseDetector

    global phrase_detector

    if not phrase_detector:
        phrase_detector = PhraseDetector(
            PHRASE_MODEL_PATH, sentences, reset_cache=True)

    return phrase_detector


# Helper function for transforming corpus
def transform_corpus(sentences: List[str]) -> List[List[str]]:
    return [get_words(sentence) for sentence in sentences]


# Helper function for loading the data
def load_data(file_path: str) -> List[JobData]:
    return load_file(file_path)


# BM25 Similarity search endpoint (POST method)
@router.post("/bm25-similarity", response_model=BM25SimilarityResult)
async def bm25_similarity(request: BM25SimilarityRequest):
    try:
        # Load data
        data: List[JobData] = load_data(request.data_file)

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
                "matched": [query for query in queries if query in result["text"]],
                "text": result["text"],
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
