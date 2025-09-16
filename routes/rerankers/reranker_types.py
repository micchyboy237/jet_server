from typing import Optional
from pydantic import BaseModel
from shared.data_types.job import JobData


class Match(BaseModel):
    score: float
    start_idx: int
    end_idx: int
    sentence: str
    text: str


class SimilarityRequest(BaseModel):
    queries: list[str]
    data_file: str = "/Users/jethroestrada/Desktop/External_Projects/Jet_Apps/my-jobs/saved/jobs.json"


class SimilarityData(BaseModel):
    id: str  # Document ID
    text: str  # The document's content/text
    score: float  # Normalized similarity score
    similarity: Optional[float]  # Raw BM25 similarity score
    matched: dict[str, int]  # Query match counts
    matched_sentences: dict[str, list[Match]]  # Query to sentence matches


class SimilarityResult(BaseModel):
    count: int
    data: list[SimilarityData]
