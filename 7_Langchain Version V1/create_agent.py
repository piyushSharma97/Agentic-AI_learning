# ============================================================
# LANGCHAIN V1: CREATE AGENT
# ============================================================
#
# create_agent() builds an agent from a model, a list of tools,
# and a system prompt. The agent decides when to call a tool,
# runs it, and then writes the final answer.
#
# Uncomment the example you want inside main().
# ============================================================

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

# Load API keys from .env. The OpenAI key is checked inside the example.
load_dotenv()


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def example_1_math_agent():
    # openai:gpt-4.1-mini is an OpenAI model, so this demo needs that key.
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing from .env")

    print("\n========== EXAMPLE 1: MATH AGENT ==========")

    # create_agent() registers the add tool and the system prompt
    # with the chat model. The model can call add when the question
    # needs arithmetic.
    math_agent = create_agent(
        model="openai:gpt-4.1-mini",
        tools=[add],
        system_prompt="You are a helpful mathematical assistant."
    )

    # invoke() sends one user message and returns the message history.
    # The agent may call add before it writes the final reply.
    math_agent_result = math_agent.invoke({
        "messages": [
            {"role": "user", "content": "What is 25 + 35?"}
        ]
    })

    # The last message is the agent's final answer.
    print(
        "[Example 1 | math agent] Final reply:",
        math_agent_result["messages"][-1].content
    )


def main():

    print("\n============================================")
    print("LANGCHAIN V1 - CREATE AGENT PRACTICE")
    print("============================================")

    # Uncomment the examples you want to execute.

    # example_1_math_agent()

    print("\n============================================")
    print("EXECUTION COMPLETED")
    print("============================================")


# Python entry point.
# This ensures main() runs only when this file
# is executed directly, not when imported elsewhere.

if __name__ == "__main__":
    main()
