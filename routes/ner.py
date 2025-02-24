from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from jet.vectors.ner import load_nlp_pipeline, extract_entities_from_text
import json

router = APIRouter()

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


class LoadEntitiesResult(BaseModel):
    id: str
    text: str
    entities: List[Entity]


class LoadEntitiesResponse(BaseModel):
    model: str
    labels: List[str]
    data: List[LoadEntitiesResult]


class ProcessedTextResponse(BaseModel):
    text: str
    entities: List[Entity]


@router.get("/entities", response_model=LoadEntitiesResponse)
def load_entities(file: str):
    """Loads entities from a specified JSON file."""
    try:
        # Load the content of the file
        file_path = Path(file)
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Prepare response in the required format
        response = {
            "model": data["model"],
            "labels": data["labels"],
            "count": len(data["results"]),
            "data": data["results"]
        }

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {str(e)}")


async def entity_generator(request: ProcessRequest):
    nlp = load_nlp_pipeline(request.model, request.labels,
                            request.style, request.chunk_size)

    for item in request.data:
        entities = extract_entities_from_text(nlp, item.text)
        response_data = ProcessedTextResponse(
            text=item.text, entities=entities)
        yield json.dumps(response_data.dict()) + "\n"


@router.post("/extract-entity", response_model=List[Entity])
def extract_entity(request: SingleTextRequest):
    nlp = load_nlp_pipeline(request.model, request.labels,
                            request.style, request.chunk_size)
    return extract_entities_from_text(nlp, request.text)


@router.post("/extract-entities")
def extract_entities(request: ProcessRequest):
    return StreamingResponse(entity_generator(request), media_type="application/json")
