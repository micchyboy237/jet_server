from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
import spacy
import uvicorn

router = APIRouter()

# Global cache for storing the loaded pipeline
nlp_cache = None


def load_nlp_pipeline(model: str, labels: List[str], style: str, chunk_size: int):
    global nlp_cache
    if nlp_cache is None:
        custom_spacy_config = {
            "gliner_model": model,
            "chunk_size": chunk_size,
            "labels": labels,
            "style": style
        }
        nlp_cache = spacy.blank("en")
        nlp_cache.add_pipe("gliner_spacy", config=custom_spacy_config)
    return nlp_cache


def merge_dot_prefixed_words(text: str) -> str:
    tokens = text.split()
    merged_tokens = []
    for i, token in enumerate(tokens):
        if token.startswith(".") and merged_tokens and not merged_tokens[-1].startswith("."):
            merged_tokens[-1] += token
        elif merged_tokens and merged_tokens[-1].endswith("."):
            merged_tokens[-1] += token
        else:
            merged_tokens.append(token)
    return " ".join(merged_tokens)


def get_unique_entities(entities: List[Dict]) -> List[Dict]:
    best_entities = {}
    for entity in entities:
        text = entity["text"]
        words = [t.replace(" ", "") for t in text.split(" ") if t]
        normalized_text = " ".join(words)
        label = entity["label"]
        score = float(entity["score"])
        entity["text"] = normalized_text
        key = f"{label}-{str(normalized_text)}"
        if key not in best_entities or score > float(best_entities[key]["score"]):
            entity["score"] = score
            best_entities[key] = entity
    return list(best_entities.values())


def extract_entities_from_text(nlp, text: str) -> List[Dict]:
    doc = nlp(text)
    return get_unique_entities([
        {
            "text": merge_dot_prefixed_words(entity.text),
            "label": entity.label_,
            "score": float(entity._.score)
        } for entity in doc.ents
    ])

# Request Models


class TextRequest(BaseModel):
    text: str


class ProcessRequest(BaseModel):
    model: str = "urchade/gliner_small-v2.1"
    labels: List[str]
    style: str = "ent"
    data: List[TextRequest]
    chunk_size: int = 250


class SingleTextRequest(BaseModel):
    text: str
    model: str = "urchade/gliner_small-v2.1"
    labels: List[str] = []
    style: str = "ent"
    chunk_size: int = 250

# Response Models


class Entity(BaseModel):
    text: str
    label: str
    score: float


class ProcessedTextResponse(BaseModel):
    text: str
    entities: List[Entity]


class ProcessResponse(BaseModel):
    data: List[ProcessedTextResponse]


@router.post("/extract-entity", response_model=List[Entity])
def extract_entity(request: SingleTextRequest):
    nlp = load_nlp_pipeline(request.model, request.labels,
                            request.style, request.chunk_size)
    return extract_entities_from_text(nlp, request.text)


@router.post("/extract-entities", response_model=ProcessResponse)
def extract_entities(request: ProcessRequest):
    results = []
    nlp = load_nlp_pipeline(request.model, request.labels,
                            request.style, request.chunk_size)

    for item in request.data:
        entities = extract_entities_from_text(nlp, item.text)
        results.append(ProcessedTextResponse(
            text=item.text, entities=entities))

    return ProcessResponse(data=results)
