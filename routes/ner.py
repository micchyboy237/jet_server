import json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from jet.vectors.ner import extract_named_entities
from typing import List, Dict
from pathlib import Path

router = APIRouter()

# Assuming the NER_MODEL, NER_LABELS, NER_STYLE constants are imported from jet.vectors.ner


class TextRequest(BaseModel):
    text: str
    model: str = "urchade/gliner_small-v2.1"  # default model value
    style: str = "default"  # adjust as needed
    labels: List[str] = ["role", "application",
                         "technology stack", "qualifications"]  # default labels


class Entity(BaseModel):
    text: str
    label: str
    score: float


class Result(BaseModel):
    id: str
    text: str
    entities: List[Entity]


class LoadEntitiesResponse(BaseModel):
    model: str
    labels: List[str]
    data: List[Result]


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


@router.post("/extract-entities")
def extract_entities(request: TextRequest):
    """API endpoint to extract named entities from the provided text."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Assuming extract_named_entities is properly defined to extract entities from the provided text
    entities = extract_named_entities([request.text])
    return {"data": entities}
