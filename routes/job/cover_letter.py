from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Dict, Optional
from jet.file.utils import save_file, load_file
from jet.logger import logger
from jet.token.token_utils import get_ollama_tokenizer
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.utils import set_global_tokenizer
from jet.llm.ollama.base import Ollama
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from jet.scrapers.utils import clean_text
from jet.utils.object import extract_values_by_paths
from tqdm import tqdm
import json
from pydantic import BaseModel

router = APIRouter()

JOBS_FILE = "saved/jobs.json"
COVER_LETTERS_FILE = "saved/job-cover-letters.json"

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
                "options": {"temperature": 0},
                # "max_prediction_ratio": 0.5
            },
        )

        return response


class JobCoverLetter(BaseModel):
    subject: str
    message: str


class JobCoverLetterRequest(BaseModel):
    cover_letters_file: Optional[str] = COVER_LETTERS_FILE


class JobGenerateCoverLettersRequest(BaseModel):
    query: str = DEFAULT_QUERY
    jobs: Optional[List[Dict]]
    jobs_file: Optional[str] = JOBS_FILE
    output_file: Optional[str] = COVER_LETTERS_FILE


class JobCoverLetterResponse(BaseModel):
    id: str
    link: str
    text: str
    response: Dict

# @router.post("/generate-cover-letters", response_model=List[JobCoverLetterResponse])


@router.post("/generate-cover-letters")
def generate_cover_letters(request: JobGenerateCoverLettersRequest):
    model = "llama3.1"
    chunk_size = 1024
    chunk_overlap = 128

    tokenizer = get_ollama_tokenizer(model)
    set_global_tokenizer(tokenizer)
    llm = Ollama(model=model)

    jobs = request.jobs
    if not jobs:
        jobs_file = request.jobs_file or JOBS_FILE
        jobs = load_file(jobs_file) or []

    output_file = request.output_file or COVER_LETTERS_FILE
    existing_results = load_file(output_file) or []
    existing_ids = {item['id'] for item in existing_results}
    jobs = [job for job in jobs if job['id'] not in existing_ids]

    if not jobs:
        raise HTTPException(status_code=400, detail="No new jobs to process.")

    json_attributes = ["title", "company", "salary",
                       "job_type", "hours_per_week", "details"]
    splitter = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    summarizer = Summarizer(llm=llm)

    def generate_stream():
        for job in tqdm(jobs, total=len(jobs), unit="job"):
            json_parts = extract_values_by_paths(
                job, json_attributes, is_flattened=True)
            text_parts = [
                f"Job Title: {json_parts['title']}", f"Company: {json_parts['company']}"]

            for key in ["salary", "job_type", "hours_per_week"]:
                if json_parts.get(key):
                    text_parts.append(
                        f"{key.replace('_', ' ').title()}: {json_parts[key]}")

            text_parts.append(f"Job Details:\n{json_parts['details']}")
            cleaned_text = clean_text("\n".join(text_parts))
            text_chunks = splitter.split_text(cleaned_text)

            try:
                response = summarizer.summarize(
                    request.query, text_chunks[0], JobCoverLetter, llm)
                result = JobCoverLetterResponse(
                    id=job['id'],
                    link=job.get('link', ''),
                    text=text_chunks[0],
                    response=response.dict(),
                )
                save_file([result.dict()] + existing_results, output_file)
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
