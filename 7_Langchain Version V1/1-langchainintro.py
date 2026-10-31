### Langchain Version V1

### Agents
# create_agent() builds an agent that can decide when to call tools.
# Each example below builds its own agent, so one demo does not
# share conversation state with the next.
#
# Uncomment the example you want inside main().


# ============================================================
# 1. IMPORTS AND ENVIRONMENT
# ============================================================

import os

import langchain
from dotenv import load_dotenv
from langchain.agents import create_agent

# Load API keys from .env. Each example checks its own key
# only when that example actually runs.
load_dotenv()


# ============================================================
# 2. TOOLS
# ============================================================

def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."


def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b


def _require_openai_key():
    """gpt-5 is an OpenAI model, so this demo needs OPENAI_API_KEY."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is missing from .env")


# ============================================================
# EXAMPLE 1: WEATHER QUESTION
# ============================================================

def example_1_weather_question():
    _require_openai_key()

    print("\n========== EXAMPLE 1: WEATHER QUESTION ==========")

    # create_agent() connects the model, the tools, and the system prompt.
    # model="gpt-5" selects the chat model. The agent can call either tool.
    weather_like_agent = create_agent(
        model="gpt-5",
        tools=[get_weather, calculate_sum],
        system_prompt="You are a helpful assistant."
    )

    # invoke() sends the user message and runs the agent loop.
    # The result dictionary holds the full message history.
    weather_like_result = weather_like_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "What is the weather like in New York?"
            }
        ]
    })

    print("[Example 1 | weather question] Messages:", weather_like_result["messages"])


# ============================================================
# EXAMPLE 2: WEATHER AGAIN
# ============================================================

def example_2_weather_again():
    _require_openai_key()

    print("\n========== EXAMPLE 2: WEATHER AGAIN ==========")

    # A second agent, built the same way, so this run is independent.
    weather_in_agent = create_agent(
        model="gpt-5",
        tools=[get_weather, calculate_sum],
        system_prompt="You are a helpful assistant."
    )

    # Same kind of weather question, with slightly different wording.
    weather_in_result = weather_in_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in New York?"
            }
        ]
    })

    print("[Example 2 | weather again] Messages:", weather_in_result["messages"])


# ============================================================
# EXAMPLE 3: WEATHER PLUS SUM
# ============================================================

def example_3_weather_plus_sum():
    _require_openai_key()

    print("\n========== EXAMPLE 3: WEATHER PLUS SUM ==========")

    # This question needs both tools: get_weather and calculate_sum.
    weather_and_sum_agent = create_agent(
        model="gpt-5",
        tools=[get_weather, calculate_sum],
        system_prompt="You are a helpful assistant."
    )

    weather_and_sum_result = weather_and_sum_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in Delhi and what is 25 + 35?"
            }
        ]
    })

    # The last message is the agent's final reply after the tool calls.
    print(
        "[Example 3 | weather plus sum] Final reply:",
        weather_and_sum_result["messages"][-1].content
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n============================================")
    print("LANGCHAIN V1 - AGENTS INTRO PRACTICE")
    print("============================================")
    print("[LangChain version] Installed version:", langchain.__version__)

    # Uncomment the examples you want to execute.

    # example_1_weather_question()

    # example_2_weather_again()

    # example_3_weather_plus_sum()

    print("\n============================================")
    print("EXECUTION COMPLETED")
    print("============================================")


# Python entry point.
# This ensures main() runs only when this file
# is executed directly, not when imported elsewhere.

if __name__ == "__main__":
    main()
