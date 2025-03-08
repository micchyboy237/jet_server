import json
import asyncio
import os
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import httpx
from llama_index.core.evaluation.dataset_generation import DatasetGenerator
from llama_index.core.evaluation import FaithfulnessEvaluator, EvaluationResult
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, Response
from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
from jet.llm.ollama import initialize_ollama_settings

# Initialize settings and FastAPI router
initialize_ollama_settings()
router = APIRouter()

# Default configurations
DEFAULT_MODEL = "llama3.1"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 256
DEFAULT_CHUNK_SIZE = 512
# Change this to your actual default path
DEFAULT_DOCUMENTS_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/my-jobs/saved/jobs.json"


# Define the input model for queries
class QueryRequest(BaseModel):
    query: str
    model: str = Field(
        DEFAULT_MODEL, description="LLM model to use for processing")
    temperature: float = Field(
        DEFAULT_TEMPERATURE, description="Temperature for LLM response generation")
    max_tokens: int = Field(
        DEFAULT_MAX_TOKENS, description="Maximum number of tokens to generate")
    chunk_size: int = Field(
        DEFAULT_CHUNK_SIZE, description="Chunk size for text splitting")
    documents_path: str = Field(
        DEFAULT_DOCUMENTS_PATH, description="Path to the document directory")


# Helper function for displaying evaluation results
def display_eval_df(query: str, response: Response, eval_result: EvaluationResult) -> None:
    display_jet_source_nodes(query, response.source_nodes)

    logger.newline()
    logger.info("Eval Results:")
    items = [(key, result) for key,
             result in eval_result.model_dump().items() if result is not None]
    for key, result in items:
        if key == 'passing':
            logger.log(f"{key.title()}:", "Passed" if result else "Failed", colors=[
                       "DEBUG", "SUCCESS" if result else "ERROR"])
        elif key == 'invalid_result':
            logger.log(f"{key.title()}:", "Valid" if not result else "Invalid", colors=[
                       "DEBUG", "SUCCESS" if not result else "ERROR"])
        else:
            logger.log(f"{key.title()}:", result, colors=["DEBUG", "SUCCESS"])


# Function to load documents dynamically
def load_documents(documents_path: str):
    if not os.path.exists(documents_path):
        raise HTTPException(
            status_code=400, detail=f"Documents path '{documents_path}' does not exist.")
    return SimpleDirectoryReader(documents_path, required_exts=[".md"]).load_data()


# Endpoint to evaluate a query with the vector index
@router.post("/evaluate/")
async def evaluate_query(request: QueryRequest):
    try:
        # Load documents from the specified path
        documents = load_documents(request.documents_path)

        # Configure the vector index with dynamic chunk size
        splitter = SentenceSplitter(chunk_size=request.chunk_size)
        vector_index = VectorStoreIndex.from_documents(
            documents, transformations=[splitter])

        # Instantiate LLM and Evaluator
        llm = Ollama(temperature=request.temperature, model=request.model)
        evaluator = FaithfulnessEvaluator(llm=llm)

        query_engine = vector_index.as_query_engine()
        response_vector = query_engine.query(request.query)
        eval_result = evaluator.evaluate_response(response=response_vector)

        # Display results
        display_eval_df(request.query, response_vector, eval_result)

        return {
            "query": request.query,
            "response": eval_result.response,
            "passing": eval_result.passing,
            "score": eval_result.score,
            "feedback": eval_result.feedback,
            "contexts": eval_result.contexts,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error evaluating query: {str(e)}")


# Endpoint to generate evaluation questions with optional parameters
@router.get("/generate_questions/")
async def generate_questions(
    num_questions_per_chunk: int = Query(
        3, description="Number of questions per document chunk"),
    chunk_size: int = Query(
        DEFAULT_CHUNK_SIZE, description="Chunk size for document splitting"),
    documents_path: str = Query(
        DEFAULT_DOCUMENTS_PATH, description="Path to the document directory")
):
    try:
        # Load documents from the specified path
        documents = load_documents(documents_path)

        question_gen_query = f"You are a Job Employer. Your task is to set up {num_questions_per_chunk} questions for an upcoming interview. The questions should be relevant to the document."

        question_generation_prompt = """\
        Context information is below.
        ---------------------
        {context_str}
        ---------------------
        Given the context information and not prior knowledge.
        Generate only questions based on the below query.
        Query: {query_str}
        """
        question_generation_template = PromptTemplate(
            question_generation_prompt)

        # Configure splitter with dynamic chunk size
        splitter = SentenceSplitter(chunk_size=chunk_size)
        vector_index = VectorStoreIndex.from_documents(
            documents, transformations=[splitter])

        question_generator = DatasetGenerator.from_documents(
            documents,
            num_questions_per_chunk=num_questions_per_chunk,
            question_gen_query=question_gen_query,
            text_question_template=question_generation_template,
        )
        eval_questions = question_generator.generate_questions_from_nodes()

        return {"generated_questions": eval_questions}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating questions: {str(e)}")


# Async function to evaluate multiple queries in bulk
async def evaluate_bulk_queries(query_engine, questions, evaluator):
    async with httpx.AsyncClient() as client:
        total_correct = 0
        total = 0
        results = []

        for question in questions:
            try:
                response = query_engine.query(question)
                eval_response = evaluator.evaluate_response(response=response)
                eval_result = 1 if eval_response.passing else 0
                total_correct += eval_result
                total += 1

                results.append({
                    "question": question,
                    "response": eval_response.response,
                    "correct": total_correct,
                    "total": total,
                    "passing": eval_response.passing,
                    "contexts": eval_response.contexts,
                    "feedback": eval_response.feedback,
                    "score": eval_response.score,
                })

            except Exception as e:
                results.append({"question": question, "error": str(e)})

        return results


# Endpoint to evaluate multiple queries in bulk
@router.post("/evaluate_bulk/")
async def evaluate_bulk(request: QueryRequest):
    try:
        # Load documents from the specified path
        documents = load_documents(request.documents_path)

        # Configure the vector index with dynamic chunk size
        splitter = SentenceSplitter(chunk_size=request.chunk_size)
        vector_index = VectorStoreIndex.from_documents(
            documents, transformations=[splitter])
        query_engine = vector_index.as_query_engine()

        # Instantiate LLM and Evaluator
        llm = Ollama(temperature=request.temperature, model=request.model)
        evaluator = FaithfulnessEvaluator(llm=llm)

        eval_results = await evaluate_bulk_queries(query_engine, [request.query], evaluator)
        return {"evaluation_results": eval_results}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error evaluating bulk queries: {str(e)}")

# Run the FastAPI app with: uvicorn app_name:app --reload
