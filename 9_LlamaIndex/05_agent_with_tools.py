# ============================================================
# LLAMAINDEX 105: AGENTIC RAG & TOOLS (ReAct Agent)
# ============================================================
#
# What is Agentic RAG?
# --------------------
# Standard RAG is a rigid pipeline: User Question -> Search Docs -> LLM Answer.
# But what if the question requires:
# - Querying multiple documents conditionally?
# - Calculating numbers based on retrieved facts?
# - Deciding that NO document retrieval is needed?
#
# An Agent in LlamaIndex uses a reasoning loop (like ReAct:
# Reasoning + Acting) to decide:
# 1. Which tool to call (e.g. Handbook Search vs. Calculator).
# 2. What parameters to pass to the tool.
# 3. How to synthesize the tool's output into a final response.
#
# Core Tools in LlamaIndex:
# - `QueryEngineTool`: Wraps any LlamaIndex QueryEngine into a tool.
# - `FunctionTool`: Wraps any standard Python function into a tool.
# ============================================================

import asyncio
import os
from pathlib import Path

import config  # loads the repo-root .env before LlamaIndex reads the keys

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI


# ------------------------------------------------------------
# 1. DEFINE A PYTHON FUNCTION TOOL (Calculator)
# ------------------------------------------------------------
def calculate_home_office_budget(team_size: int, include_internet_months: int = 12) -> float:
    """
    Calculates the total first-year home office budget for a given number of employees.
    At TechFlow:
    - One-time setup stipend: $1,500 per employee.
    - Monthly internet reimbursement: $100/month per employee.
    """
    setup_cost_per_person = 1500.0
    internet_per_month_per_person = 100.0

    total_setup = team_size * setup_cost_per_person
    total_internet = team_size * (internet_per_month_per_person * include_internet_months)

    return total_setup + total_internet


def create_agent():
    """Builds and returns a ReAct agent with both Document RAG and Python tools."""
    
    # 1. Load handbook and create QueryEngine
    data_dir = Path(__file__).parent / "data"
    documents = SimpleDirectoryReader(input_dir=str(data_dir)).load_data()
    index = VectorStoreIndex.from_documents(documents)
    handbook_query_engine = index.as_query_engine(similarity_top_k=2)

    # 2. Wrap QueryEngine as an Agent Tool
    # Note: The description is CRITICAL. The LLM reads the description to
    # decide whether this tool should be selected for a given prompt.
    handbook_tool = QueryEngineTool(
        query_engine=handbook_query_engine,
        metadata=ToolMetadata(
            name="company_handbook",
            description=(
                "Useful for answering any questions about TechFlow Innovations company policies, "
                "leave guidelines, founders, locations, PTO, and benefits."
            ),
        ),
    )

    # 3. Wrap Python function as an Agent Tool
    calculator_tool = FunctionTool.from_defaults(
        fn=calculate_home_office_budget,
        name="budget_calculator",
        description=(
            "Useful for calculating total home office and internet reimbursement costs "
            "for a given team size over a specified number of months."
        ),
    )

    # 4. Initialize LLM (gpt-4o-mini is fast and follows tool schema well)
    llm = OpenAI(model="gpt-4o-mini", temperature=0)

    # Current ReActAgent is constructed directly. from_tools() and .chat()
    # belong to the older agent API and are not on this class.
    # verbose=True prints each Thought / Action / Observation step.
    agent = ReActAgent(
        tools=[handbook_tool, calculator_tool],
        llm=llm,
        verbose=True,
    )
    return agent


async def ask(agent, question: str) -> None:
    """Run one user message and print the assistant text."""
    result = await agent.run(user_msg=question)
    message = getattr(result, "response", result)
    text = getattr(message, "content", message)
    print(f"\nFinal Agent Answer:\n{text}\n")


async def example_handbook_only(agent) -> None:
    """
    The handbook tool is enough. The agent should search the document
    and leave the calculator unused.
    """
    print("\n" + "=" * 65)
    print("TASK 1: Document search only")
    print("=" * 65)
    await ask(agent, "What are the core collaboration hours at TechFlow?")


async def example_handbook_and_calculator(agent) -> None:
    """
    The stipend amounts live in the handbook, and the total is a calculation.
    A useful agent calls the handbook, then the calculator. This prompt
    names the calculator so the lesson shows that tool path.
    """
    print("\n" + "=" * 65)
    print("TASK 2: Handbook plus calculator")
    print("=" * 65)
    await ask(
        agent,
        "If I am hiring a team of 5 engineers, what will be their total first-year "
        "home office and internet budget? Use the budget calculator tool.",
    )


async def run_examples(agent) -> None:
    """One event loop for every task. Comment a line to skip that task."""
    await example_handbook_only(agent)
    await example_handbook_and_calculator(agent)


def main():
    print("=" * 65)
    print("LLAMAINDEX INTRO PRACTICE: AGENTIC RAG & TOOLS")
    print("=" * 65)

    if not os.getenv("OPENAI_API_KEY"):
        print("[WARNING] Set OPENAI_API_KEY in .env to run this agent example.")
        return

    agent = create_agent()
    asyncio.run(run_examples(agent))


if __name__ == "__main__":
    main()

