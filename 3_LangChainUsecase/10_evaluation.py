"""
=============================================================================
USE CASE 10: EVALUATION IN LANGCHAIN
=============================================================================
In production AI engineering, you cannot improve what you cannot measure.
Prompt changes, model swaps, and chunking tuning require quantitative evaluation
to detect regressions and hallucinations before shipping to users.

Key Concepts Covered:
1. LLM-as-a-Judge paradigm
2. Core Evaluation Metrics:
   - Correctness / Factual Accuracy (vs Ground Truth)
   - Faithfulness / Groundedness (Grounded in context vs Hallucinated)
   - Answer Relevance (Does it satisfy the user's prompt?)
3. Structured Evaluation Schema using Pydantic
4. Test Dataset Batch Runner & Scorecard Summary
=============================================================================
"""

from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

# Import shared LLM config
from config import get_llm, safe_execute


# ---------------------------------------------------------------------------
# 1. Define Evaluation Metrics Schema
# ---------------------------------------------------------------------------
class EvaluationResult(BaseModel):
    factual_accuracy_score: int = Field(
        description="Rating from 1 (completely wrong) to 5 (fully accurate) comparing against ground truth.",
        ge=1, le=5
    )
    faithfulness_score: int = Field(
        description="Rating from 1 to 5: Is the answer supported by evidence without hallucinated facts?",
        ge=1, le=5
    )
    relevance_score: int = Field(
        description="Rating from 1 to 5: Does the response directly address the question without fluff?",
        ge=1, le=5
    )
    verdict: Literal["PASS", "FAIL"] = Field(
        description="'PASS' if average score is >= 4 and no critical errors exist, else 'FAIL'."
    )
    reasoning: str = Field(
        description="Clear explanation explaining the rationale for the given scores."
    )


# ---------------------------------------------------------------------------
# 2. Build the Evaluator (LLM-as-a-Judge)
# ---------------------------------------------------------------------------
def create_evaluator_chain():
    llm = get_llm(temperature=0.0) # Zero temperature for deterministic evaluation
    evaluator_llm = llm.with_structured_output(EvaluationResult)
    
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an impartial, rigorous AI Evaluation Judge. "
            "Evaluate the candidate AI Response against the Ground Truth and Question. "
            "Score on Factual Accuracy, Faithfulness, and Relevance from 1 to 5."
        )),
        ("human", (
            "Question: {question}\n\n"
            "Ground Truth Reference:\n{ground_truth}\n\n"
            "Candidate AI Response to Evaluate:\n{candidate_response}"
        ))
    ])
    
    return eval_prompt | evaluator_llm


# ---------------------------------------------------------------------------
# 3. Batch Evaluation on Test Dataset
# ---------------------------------------------------------------------------
def demo_batch_evaluation():
    print("--- 1. LLM-as-a-Judge Batch Evaluation ---")
    evaluator = create_evaluator_chain()
    
    # Test dataset with realistic positive and flawed candidate outputs
    test_cases = [
        {
            "test_id": "TEST-01 (High Quality Answer)",
            "question": "What is Python's GIL and why does it exist?",
            "ground_truth": (
                "The Global Interpreter Lock (GIL) is a mutex in CPython that prevents multiple "
                "threads from executing Python bytecode simultaneously, designed to ensure thread safety "
                "for CPython's reference counting memory management."
            ),
            "candidate_response": (
                "Python's GIL (Global Interpreter Lock) is a synchronization lock in CPython that allows "
                "only one native thread to run Python bytecode at a time, protecting internal memory management."
            )
        },
        {
            "test_id": "TEST-02 (Hallucinated / Incorrect Answer)",
            "question": "Which company developed the PyTorch deep learning framework?",
            "ground_truth": "PyTorch was originally developed and released by Facebook's AI Research lab (FAIR, now Meta).",
            "candidate_response": "PyTorch was originally created by Google Brain in 2015 as a successor to TensorFlow."
        }
    ]
    
    results_summary = []
    
    for case in test_cases:
        print(f"\nEvaluating: {case['test_id']}")
        print(f"Question: {case['question']}")
        print(f"Candidate: \"{case['candidate_response']}\"")
        
        evaluation: EvaluationResult = evaluator.invoke({
            "question": case["question"],
            "ground_truth": case["ground_truth"],
            "candidate_response": case["candidate_response"]
        })
        
        results_summary.append({
            "id": case["test_id"],
            "accuracy": evaluation.factual_accuracy_score,
            "faithfulness": evaluation.faithfulness_score,
            "relevance": evaluation.relevance_score,
            "verdict": evaluation.verdict,
            "reasoning": evaluation.reasoning
        })
        
        print(f"Verdict: [{evaluation.verdict}]")
        print(f"Scores -> Accuracy: {evaluation.factual_accuracy_score}/5 | Faithfulness: {evaluation.faithfulness_score}/5 | Relevance: {evaluation.relevance_score}/5")
        print(f"Reasoning: {evaluation.reasoning}")
        print("-" * 60)
        
    # Print Executive Scorecard
    print("\n" + "=" * 60)
    print("EVALUATION SCORECARD SUMMARY")
    print("=" * 60)
    print(f"{'Test Case':<32} | {'Acc':<4} | {'Fth':<4} | {'Rel':<4} | {'Verdict':<6}")
    print("-" * 60)
    for r in results_summary:
        print(f"{r['id']:<32} | {r['accuracy']:<4} | {r['faithfulness']:<4} | {r['relevance']:<4} | {r['verdict']:<6}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    safe_execute("Evaluation Demonstrations", lambda: [
        demo_batch_evaluation(),
    ])


if __name__ == "__main__":
    main()

