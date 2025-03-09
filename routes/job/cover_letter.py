import os
import random
from fastapi import APIRouter, HTTPException
from fastapi.param_functions import Depends
from fastapi.responses import StreamingResponse
from typing import List, Dict, Optional
from jet.file.utils import save_file, load_file
from jet.llm.models import OLLAMA_MODEL_NAMES
from jet.logger import logger
from jet.token.token_utils import get_ollama_tokenizer
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.utils import set_global_tokenizer
from jet.llm.ollama.base import Ollama
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from jet.scrapers.utils import clean_text
from jet.utils.object import extract_values_by_paths
from shared.data_types.job import JobData
from tqdm import tqdm
import json
from pydantic import BaseModel, ValidationError

router = APIRouter()

JOBS_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"
COVER_LETTERS_FILE = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/job-cover-letters.json"

PROMPT_TEMPLATE = """\
Given the prompt, schema and not prior knowledge, answer the query.
The generated JSON must pass the provided schema when validated.
Query: {query_str}

Prompt:
```text
{prompt_str}
```
Response:
"""

MESSAGE_TEMPLATE = """
Hi <employer_name>,

I'm interested in the position as the tech stack mentioned seems to be an ideal fit for my skillset.
My primary skills <continue>.

<other_relevant_info>

Here is a link to my website with portfolio and latest resume:
https://jethro-estrada.web.app
"""

DEFAULT_QUERY = f"""
Generate a cover letter based on provided job post information. A company may be an organization or employer name. Follow instructions relevant to the subject or message if any.

You may follow this message template:
```template
{MESSAGE_TEMPLATE}
```
""".strip()


class Summarizer:
    def __init__(self, llm, verbose: bool = True, streaming: bool = False, prompt_tmpl: str = PROMPT_TEMPLATE):
        self.llm = llm
        self.verbose = verbose
        self.streaming = streaming

        self.qa_prompt = PromptTemplate(prompt_tmpl)

    def summarize(self, query: str, prompt: str, output_cls: BaseModel, llm: Ollama) -> BaseModel:
        response = llm.structured_predict(
            output_cls=output_cls,
            prompt=self.qa_prompt,
            # context_str=context,
            # schema_str=schema,
            # sample_str=DEFAULT_SAMPLE,
            query_str=query,
            prompt_str=prompt,
            llm_kwargs={
                "options": {
                    "temperature": 0.3,
                    "seed": random.randint(1, 9999)
                }
                # "max_prediction_ratio": 0.5
            },
        )

        return response


class JobCoverLetter(BaseModel):
    subject: str
    message: str


class JobCoverLetterItem(BaseModel):
    id: str
    link: str
    posted_date: str
    text: str
    response: JobCoverLetter


class JobCoverLettersData(BaseModel):
    count: int
    data: list[JobCoverLetterItem]


def get_default_file():
    return COVER_LETTERS_FILE


@router.get("/", response_model=JobCoverLettersData)
def load_cover_letters(file: str = Depends(get_default_file)):
    """Loads cover letters from a specified JSON file."""
    if not os.path.isfile(file):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Validate the data using Pydantic
    try:
        validated_data = [JobCoverLetterItem(**item) for item in raw_data]
    except ValidationError as e:
        logger.error(e)
        raise HTTPException(
            status_code=400, detail=f"Invalid data format: {e}")

    return {
        "count": len(validated_data),
        "data": validated_data
    }


@router.get("/{id}", response_model=JobCoverLetterItem)
def load_cover_letter_by_id(
    id: str, file: str = Depends(get_default_file)
):
    """Loads a single cover letter by its ID."""
    if not os.path.isfile(file):
        raise HTTPException(status_code=404, detail="File not found")

    with open(file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Validate the data using Pydantic
    validated_data = [JobCoverLetterItem(**item) for item in raw_data]

    # Find the cover letter by ID
    cover_letter = next(
        (item for item in validated_data if item.id == id), None)

    if not cover_letter:
        raise HTTPException(status_code=404, detail="Cover letter not found")

    return cover_letter


class JobCoverLetterRequest(BaseModel):
    cover_letters_file: Optional[str] = COVER_LETTERS_FILE


class JobGenerateCoverLettersRequest(BaseModel):
    job_ids: Optional[list[str]] = None
    attributes: list[str] = []
    model: OLLAMA_MODEL_NAMES = "llama3.2"
    query: str = DEFAULT_QUERY
    jobs_file: str = JOBS_FILE
    output_file: str = COVER_LETTERS_FILE


class JobCoverLetterResponse(BaseModel):
    id: str
    link: str
    posted_date: str
    text: str
    response: Dict


class JobCoverLetter(BaseModel):
    subject: str
    message: str


class CoverLetterRequest(BaseModel):
    text: Optional[str] = None
    job_id: Optional[str] = None
    attributes: list[str] = []
    query: str = DEFAULT_QUERY
    model: OLLAMA_MODEL_NAMES = "llama3.2"
    output_file: Optional[str] = COVER_LETTERS_FILE


@router.post("/generate-cover-letter", response_model=JobCoverLetter)
def generate_cover_letter(request: CoverLetterRequest):
    cover_letter_context: str = ""

    if request.text:
        cover_letter_context = request.text
    elif request.job_id:
        jobs: list[JobData] = load_file(JOBS_FILE) or []
        job = next((job for job in jobs if job['id'] == request.job_id), None)
        if job:
            attributes = request.attributes or ["title", "details"]
            json_parts_dict = extract_values_by_paths(
                job, attributes, is_flattened=True)
            text_parts = [f"{key.title().replace('_', ' ')}: {', '.join(value) if isinstance(value, list) else value}" for key,
                          value in json_parts_dict.items()]
            cover_letter_context = "\n".join(text_parts) if text_parts else ""

    tokenizer = get_ollama_tokenizer(request.model)
    set_global_tokenizer(tokenizer)
    llm = Ollama(model=request.model)
    summarizer = Summarizer(llm=llm)

    try:
        response = summarizer.summarize(
            request.query, cover_letter_context, JobCoverLetter, llm)
        cover_letter = JobCoverLetter(**response.dict())

        # Save the generated cover letter
        if job:
            output_file = request.output_file or COVER_LETTERS_FILE
            existing_results: list[JobData] = load_file(output_file) or []
            existing_results.insert(
                0, {"id": job["id"], "link": job["link"], "text": cover_letter_context, "posted_date": job['posted_date'], "response": cover_letter.dict()})
            existing_results = list(
                {job["id"]: job for job in existing_results}.values())
            sorted_results = sorted(
                existing_results, key=lambda x: x['posted_date'], reverse=True)
            save_file(sorted_results, output_file)

        return cover_letter
    except Exception as e:
        logger.error(f"Error generating cover letter: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to generate cover letter.")


@router.post("/generate-cover-letters")
def generate_cover_letters(request: JobGenerateCoverLettersRequest):
    tokenizer = get_ollama_tokenizer(request.model)
    set_global_tokenizer(tokenizer)
    llm = Ollama(model=request.model)

    jobs_file = request.jobs_file or JOBS_FILE
    jobs: list[JobData] = load_file(jobs_file) or []

    if request.job_ids:
        jobs = [job for job in jobs if job['id'] in request.job_ids]

    output_file = request.output_file or COVER_LETTERS_FILE
    existing_results: list[JobData] = load_file(output_file) or []
    existing_ids = {item['id'] for item in existing_results}
    jobs = [job for job in jobs if job['id'] not in existing_ids]

    if not jobs:
        raise HTTPException(status_code=400, detail="No new jobs to process.")

    attributes = request.attributes or ["title", "details"]

    summarizer = Summarizer(llm=llm)

    def generate_stream():
        nonlocal existing_results
        for job in tqdm(jobs, total=len(jobs), unit="job"):
            json_parts_dict = extract_values_by_paths(
                job, attributes, is_flattened=True)
            text_parts = []
            for key, value in json_parts_dict.items():
                value_str = str(value)
                if isinstance(value, list):
                    value_str = ", ".join(value)
                text_parts.append(
                    f"{key.title().replace('_', ' ')}: {value_str}")
            cover_letter_context = "\n".join(text_parts) if text_parts else ""

            try:
                response = summarizer.summarize(
                    request.query, cover_letter_context, JobCoverLetter, llm)
                result = JobCoverLetterResponse(
                    id=job['id'],
                    link=job.get('link', ''),
                    posted_date=job.get('posted_date', ''),
                    text=cover_letter_context,
                    response=response.dict(),
                )

                existing_results.insert(0, result.dict())
                existing_results = list(
                    {job["id"]: job for job in existing_results}.values())
                sorted_results = sorted(
                    existing_results, key=lambda x: x['posted_date'], reverse=True)
                save_file(sorted_results, output_file)

                yield result.json() + "\n"
            except Exception as e:
                logger.error(f"Error processing job {job['id']}: {e}")
                continue

    return StreamingResponse(generate_stream(), media_type="application/json")


@router.get("/cover-letters", response_model=List[JobCoverLetterResponse])
def get_cover_letters(request: JobCoverLetterRequest):
    cover_letters_file = request.cover_letters_file or COVER_LETTERS_FILE
    results = load_file(cover_letters_file) or []
    return results
