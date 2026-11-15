
# ============================================================
# RESEARCH AGENT
# Demonstrating TypedDict + dataclass + Pydantic
# ============================================================

# -------------------- IMPORTS --------------------

from typing_extensions import TypedDict
# TypedDict: Defines the structure of shared LangGraph state.

from dataclasses import dataclass
# dataclass: Creates structured Python objects with defaults.

from pydantic import BaseModel, Field
# BaseModel and Field: Validate incoming user data.

from langgraph.graph import StateGraph, START, END
# StateGraph: Builds the workflow.
# START and END: Define graph entry and exit.

from langgraph.runtime import Runtime
# Runtime: Gives nodes access to runtime context.


# ============================================================
# 1. PYDANTIC: VALIDATE USER INPUT
# ============================================================

class ResearchRequest(BaseModel):

    # User must provide a question with at least 10 characters.
    question: str = Field(min_length=10)

    # Number of search results must be between 1 and 5.
    max_results: int = Field(default=3, ge=1, le=5)


# WHY PYDANTIC?
# User input is external and potentially invalid.
# We want to reject invalid input before starting the graph.


# ============================================================
# 2. TYPEDDICT: DEFINE LANGGRAPH SHARED STATE
# ============================================================

class ResearchState(TypedDict):

    question: str
    search_results: list[str]
    answer: str


# WHY TYPEDDICT?
# This is the shared data passed between graph nodes.
# Every node can read existing state and return partial updates.


# ============================================================
# 3. DATACLASS: DEFINE RUNTIME CONFIGURATION
# ============================================================

@dataclass
class ResearchConfig:

    # Maximum number of results to retrieve.
    max_results: int = 3

    # Search provider configuration.
    search_provider: str = "mock_search"


# WHY DATACLASS?
# Configuration is separate from the data flowing through
# the graph. It contains settings required during execution.
# We don't need Pydantic validation for this simple config.


# ============================================================
# 4. NODE 1: SEARCH NODE
# ============================================================

def search_node(
    state: ResearchState,
    runtime: Runtime[ResearchConfig]
) -> dict:

    # Read question from shared state.
    question = state["question"]

    # Read configuration from runtime context.
    config = runtime.context

    print("\nSearching for:", question)
    print("Provider:", config.search_provider)
    print("Maximum results:", config.max_results)

    # Simulated search results.
    all_results = [
        "LangGraph is a framework for building stateful agents.",
        "LangGraph uses nodes, edges and shared state.",
        "LangGraph supports conditional routing and loops.",
        "LangGraph supports checkpointing and persistence.",
        "LangGraph supports human-in-the-loop workflows."
    ]

    # Apply configured result limit.
    results = all_results[:config.max_results]

    # Return only the state fields we want to update.
    return {
        "search_results": results
    }


# ============================================================
# 5. NODE 2: ANSWER NODE
# ============================================================

def answer_node(state: ResearchState) -> dict:

    # Read search results from the updated shared state.
    results = state["search_results"]

    # Generate a simple answer from search results.
    answer = (
        f"Your question was: {state['question']}\n\n"
        "Research findings:\n"
        + "\n".join(results)
    )

    return {
        "answer": answer
    }


# ============================================================
# 6. BUILD LANGGRAPH WORKFLOW
# ============================================================

builder = StateGraph(
    state_schema=ResearchState,
    context_schema=ResearchConfig
)

# Register nodes.
builder.add_node("search", search_node)
builder.add_node("answer", answer_node)

# Define execution flow.
builder.add_edge(START, "search")
builder.add_edge("search", "answer")
builder.add_edge("answer", END)

# Compile graph.
graph = builder.compile()


# ============================================================
# 7. EXECUTE THE AGENT
# ============================================================

def main():

    # External input.
    raw_input = {
        "question": "What is LangGraph and how does it work?",
        "max_results": 3
    }

    try:

        # STEP 1: Validate using Pydantic.
        request = ResearchRequest(**raw_input)

        print("Input validation successful!")

        # STEP 2: Prepare graph state.
        initial_state: ResearchState = {
            "question": request.question,
            "search_results": [],
            "answer": ""
        }

        # STEP 3: Prepare runtime configuration.
        config = ResearchConfig(
            max_results=request.max_results,
            search_provider="mock_search"
        )

        # STEP 4: Invoke LangGraph.
        result = graph.invoke(
            initial_state,
            context=config
        )

        # STEP 5: Display final result.
        print("\n========== FINAL ANSWER ==========")
        print(result["answer"])

    except Exception as error:
        print("Error:", error)


if __name__ == "__main__":
    main()
