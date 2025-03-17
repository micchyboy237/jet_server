from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

# from llm.eval.converted_notebooks.evaluation.faithfulness_eval import FaithfulnessEvaluator, evaluate_query_engine
# from llm.eval.converted_notebooks.evaluation.pairwise_eval import PairwiseComparisonEvaluator, display_eval_df
# from llm.eval.converted_notebooks.evaluation.answer_and_context_relevancy import AnswerConsistencyEvaluator, evaluate_results, EvalJudges

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

# Request model for single evaluation (faithfulness)


class FaithfulnessEvaluationRequest(BaseModel):
    system: str
    query: str
    result: str
    gold_reference: str
    model: str

# Response model for evaluation (faithfulness)


class FaithfulnessEvaluationResponse(BaseModel):
    system: str
    model: str
    query: str
    similarity_score: float
    is_accurate: bool

# Request model for pairwise comparison


class PairwiseComparisonRequest(BaseModel):
    system: str
    query: str
    results: Dict[str, str]  # {model_name: result}
    gold_reference: str

# Response model for pairwise comparison


class PairwiseComparisonResponse(BaseModel):
    system: str
    query: str
    comparisons: List[FaithfulnessEvaluationResponse]

# Request model for answer and context relevancy evaluation


class AnswerContextRelevancyRequest(BaseModel):
    system: str
    query: str
    results: Dict[str, str]  # {model_name: result}
    gold_reference: str

# Response model for answer and context relevancy evaluation


class AnswerContextRelevancyResponse(BaseModel):
    system: str
    query: str
    relevancy_scores: List[float]
    is_accurate: bool


@router.post("/evaluate_faithfulness", response_model=FaithfulnessEvaluationResponse)
async def evaluate_faithfulness(request: FaithfulnessEvaluationRequest):
    """
    Evaluates a single model output using the faithfulness evaluation function.
    """
    # Use the FaithfulnessEvaluator for evaluation
    evaluator = FaithfulnessEvaluator(model=request.model)
    similarity_score = await evaluate_query_engine(request.result, [request.gold_reference])
    is_accurate = similarity_score > 0.7  # Example threshold for faithfulness

    return FaithfulnessEvaluationResponse(
        system=request.system,
        model=request.model,
        query=request.query,
        similarity_score=similarity_score,
        is_accurate=is_accurate
    )


@router.post("/compare_pairwise", response_model=PairwiseComparisonResponse)
async def compare_pairwise(request: PairwiseComparisonRequest):
    """
    Compares multiple model responses using pairwise comparison evaluation.
    """
    # Create the evaluator
    evaluator = PairwiseComparisonEvaluator(
        model_list=list(request.results.keys()))

    comparisons = []
    for model, result in request.results.items():
        eval_result = evaluator.evaluate(
            query=request.query, response=result, reference=request.gold_reference)
        comparisons.append(FaithfulnessEvaluationResponse(
            system=request.system,
            model=model,
            query=request.query,
            similarity_score=eval_result['score'],
            is_accurate=eval_result['score'] > 0.7
        ))

    # Display evaluation results (e.g., logging or visualization)
    display_eval_df(request.query, request.results, None, comparisons)

    return PairwiseComparisonResponse(
        system=request.system,
        query=request.query,
        comparisons=comparisons
    )


@router.post("/evaluate_answer_relevancy", response_model=AnswerContextRelevancyResponse)
async def evaluate_answer_relevancy(request: AnswerContextRelevancyRequest):
    """
    Evaluates multiple model responses for answer and context relevancy.
    """
    evaluator = AnswerConsistencyEvaluator()
    relevancy_scores = []

    for model, result in request.results.items():
        score = evaluator.evaluate(
            query=request.query, response=result, reference=request.gold_reference)
        relevancy_scores.append(score)

    is_accurate = all(score > 0.7 for score in relevancy_scores)

    return AnswerContextRelevancyResponse(
        system=request.system,
        query=request.query,
        relevancy_scores=relevancy_scores,
        is_accurate=is_accurate
    )
