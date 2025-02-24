from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from jet.vectors.ner import load_nlp_pipeline, extract_entities_from_text

router = APIRouter()

# Global cache for storing the loaded pipeline
nlp_cache = None


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
    labels: List[str]
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
