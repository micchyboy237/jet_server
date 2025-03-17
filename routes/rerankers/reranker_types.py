from pydantic import BaseModel
from shared.data_types.job import JobData


class SimilarityRequest(BaseModel):
    queries: list[str]
    data_file: str = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"


class SimilarityData(BaseModel):
    score: float
    similarity: float
    matched: list[str]
    result: JobData


class SimilarityResult(BaseModel):
    count: int
    data: list[SimilarityData]
