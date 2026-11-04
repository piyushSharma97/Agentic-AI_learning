"""
=============================================================================
USE CASE 2: STRUCTURED OUTPUT IN LANGCHAIN
=============================================================================
LLMs naturally output unstructured, free-form text. In production applications,
we need strictly validated data types (JSON, Pydantic objects, TypedDicts)
to feed databases, APIs, or downstream microservices.

Key Concepts Covered:
1. `model.with_structured_output(PydanticModel)` (Recommended modern approach)
2. `model.with_structured_output(TypedDict)` (Lightweight dictionary schema)
3. Enums, Nested Pydantic Models, and Field validations
4. Classic `PydanticOutputParser` with `format_instructions`
5. Real-World Extraction: Parsing an email into Action Items & Metadata
=============================================================================
"""

from typing import List, Optional, Literal
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Import shared LLM config
from config import get_llm, safe_execute


# ---------------------------------------------------------------------------
# 1. Pydantic Schemas for Structured Output
# ---------------------------------------------------------------------------
class PriorityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionItem(BaseModel):
    task: str = Field(description="The concrete actionable task to be completed.")
    assignee: Optional[str] = Field(description="Person responsible for the task, or None if unassigned.")
    deadline: Optional[str] = Field(description="Due date or time frame mentioned, or None.")
    priority: PriorityLevel = Field(description="Assessed priority level.")


class MeetingSummary(BaseModel):
    meeting_title: str = Field(description="Descriptive title of the meeting.")
    key_decisions: List[str] = Field(description="Key decisions agreed upon during the discussion.")
    action_items: List[ActionItem] = Field(description="List of action items extracted from the discussion.")
    sentiment: Literal["positive", "neutral", "contentious"] = Field(
        description="Overall sentiment of the meeting discussion."
    )


# ---------------------------------------------------------------------------
# 2. Modern with_structured_output() with Pydantic
# ---------------------------------------------------------------------------
def demo_pydantic_structured_output():
    print("--- 1. with_structured_output using Pydantic ---")
    llm = get_llm()
    
    # Bind schema directly to the model
    structured_llm = llm.with_structured_output(MeetingSummary)
    
    unstructured_notes = """
    Team Sync - Q4 Infrastructure Migration
    Attendees: Alice, Bob, Charlie.
    Alice announced that we officially decided to migrate the primary database from MySQL to PostgreSQL
    by end of November. Everyone agreed we will use AWS Aurora for hosting.
    Bob needs to write the terraform scripts by this Friday (high priority).
    Charlie will conduct security review on VPC endpoints by next Wednesday.
    Alice will prepare the rollback strategy doc before Nov 15th.
    The meeting was very collaborative and positive overall.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an executive assistant. Extract structured meeting notes from the provided text."),
        ("human", "{meeting_text}")
    ])
    
    chain = prompt | structured_llm
    result: MeetingSummary = chain.invoke({"meeting_text": unstructured_notes})
    
    print(f"Type: {type(result)}")
    print(f"Title: {result.meeting_title}")
    print(f"Sentiment: {result.sentiment}")
    print("\nKey Decisions:")
    for d in result.key_decisions:
        print(f"  - {d}")
    print("\nAction Items:")
    for item in result.action_items:
        print(f"  - Task: {item.task} | Assignee: {item.assignee} | Due: {item.deadline} | Priority: {item.priority.value}")
    print()


# ---------------------------------------------------------------------------
# 3. Modern with_structured_output() with TypedDict
# ---------------------------------------------------------------------------
def demo_typed_dict_structured_output():
    print("--- 2. with_structured_output using TypedDict ---")
    from typing_extensions import TypedDict
    
    class MovieReviewAnalysis(TypedDict):
        movie_title: str
        genres: List[str]
        rating_out_of_10: float
        is_recommended: bool
        summary_verdict: str

    llm = get_llm()
    structured_llm = llm.with_structured_output(MovieReviewAnalysis)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract structured information from the movie review."),
        ("human", "{review}")
    ])
    
    chain = prompt | structured_llm
    review_text = (
        "Christopher Nolan's 'Oppenheimer' is an intense biographical thriller that spans history and drama. "
        "Cillian Murphy gives a career-defining performance. The sound design and pacing are masterful. "
        "A definite must-watch, I give it a solid 9.2 out of 10!"
    )
    
    result = chain.invoke({"review": review_text})
    print(f"Output dictionary:\n{result}\n")


# ---------------------------------------------------------------------------
# 4. Classic PydanticOutputParser (Prompt-based formatting)
# ---------------------------------------------------------------------------
def demo_classic_pydantic_parser():
    print("--- 3. Classic PydanticOutputParser (Works on any raw text LLM) ---")
    
    class ProductComparison(BaseModel):
        product_a: str = Field(description="Name of product A")
        product_b: str = Field(description="Name of product B")
        winner: str = Field(description="Overall recommended product")
        reason: str = Field(description="Reasoning behind recommendation")
        
    parser = PydanticOutputParser(pydantic_object=ProductComparison)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a tech gadget analyst.\n{format_instructions}"),
        ("human", "Compare {product_1} and {product_2} for a college student.")
    ]).partial(format_instructions=parser.get_format_instructions())
    
    print("Injected Format Instructions sample:")
    print(parser.get_format_instructions()[:200] + "...\n")
    
    llm = get_llm()
    chain = prompt | llm | parser
    
    result = chain.invoke({"product_1": "MacBook Air M2", "product_2": "Dell XPS 13"})
    print(f"Parsed Object:\nProduct A: {result.product_a}\nProduct B: {result.product_b}\nWinner: {result.winner}\nReason: {result.reason}\n")


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    safe_execute("Structured Output Demonstrations", lambda: [
        demo_pydantic_structured_output(),
        demo_typed_dict_structured_output(),
        demo_classic_pydantic_parser(),
    ])


if __name__ == "__main__":
    main()

