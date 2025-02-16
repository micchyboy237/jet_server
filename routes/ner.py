import spacy
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from jet.vectors.ner import NER_LABELS, NER_MODEL, NER_STYLE, extract_named_entities
from typing import List, Dict

router = APIRouter()


class TextRequest(BaseModel):
    text: str
    model: str = NER_MODEL
    style: str = NER_STYLE
    labels: List[str] = NER_LABELS


@router.post("/extract-entities")
def extract_entities(request: TextRequest):
    """API endpoint to extract named entities from the provided text."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    entities = extract_named_entities([request.text])
    return {"data": entities}
