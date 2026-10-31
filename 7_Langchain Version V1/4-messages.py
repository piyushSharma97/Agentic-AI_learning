# ============================================================
# LANGCHAIN V1: MESSAGE TYPES
# ============================================================
#
# Examples covered:
#
# 1. Plain string prompt
# 2. Text prompt
# 3. SystemMessage + HumanMessage
# 4. Detailed system instructions
# 5. Message metadata
# 6. AIMessage conversation history
# 7. ToolMessage
#
# Each example builds its own Groq chat model.
# Uncomment the example you want inside main().
# ============================================================


# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# Load API keys from the .env file.
# The Groq key is checked inside each example, not at import time.
load_dotenv()


def _require_groq_key():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY is missing from .env")


def _groq_chat_model():
    # The provider:model format tells LangChain which integration
    # and model to use.
    return init_chat_model("groq:openai/gpt-oss-120b")


# ============================================================
# EXAMPLE 1: PLAIN STRING
# ============================================================

def example_1_plain_string():
    _require_groq_key()

    print("\n========== EXAMPLE 1: PLAIN STRING ==========")

    plain_string_model = _groq_chat_model()

    # Send a simple text prompt to the model.
    # A string is accepted as shorthand for a HumanMessage.
    plain_string_response = plain_string_model.invoke(
        "Please explain Artificial Intelligence."
    )

    print(
        "[Example 1 | plain string] Reply text:",
        plain_string_response.content
    )


# ============================================================
# EXAMPLE 2: TEXT PROMPT
# ============================================================

def example_2_text_prompt():
    _require_groq_key()

    print("\n========== EXAMPLE 2: TEXT PROMPT ==========")

    # A plain string can be passed directly to invoke().
    #
    # LangChain treats this string as a user message.
    #
    # Text prompts are useful when:
    # 1. You have a single, standalone question.
    # 2. You don't need to maintain conversation history.
    # 3. You want a simple interaction with an LLM.

    text_prompt_model = _groq_chat_model()

    text_prompt_response = text_prompt_model.invoke("What is LangChain?")

    print(
        "[Example 2 | text prompt] Reply text:",
        text_prompt_response.content
    )


# ============================================================
# EXAMPLE 3: SYSTEM MESSAGE AND HUMAN MESSAGE
# ============================================================

def example_3_system_and_human():
    _require_groq_key()

    print("\n========== EXAMPLE 3: SYSTEM + HUMAN ==========")

    system_human_model = _groq_chat_model()

    # SystemMessage:
    # Defines the behaviour, role and instructions of the LLM.
    poetry_system_message = SystemMessage(
        content="You are a poetry expert."
    )

    # HumanMessage:
    # Represents the input provided by the user.
    poetry_human_message = HumanMessage(
        content="Write a poem about Artificial Intelligence."
    )

    # Create a conversation using multiple messages.
    poetry_messages = [
        poetry_system_message,
        poetry_human_message
    ]

    # Send the complete message list to the model.
    system_human_response = system_human_model.invoke(poetry_messages)

    # The model returns an AIMessage object.
    print(
        "[Example 3 | system + human] Reply text:",
        system_human_response.content
    )


# ============================================================
# EXAMPLE 4: SYSTEM MESSAGE WITH DETAILED INSTRUCTIONS
# ============================================================

def example_4_detailed_system():
    _require_groq_key()

    print("\n========== EXAMPLE 4: DETAILED SYSTEM ==========")

    detailed_system_model = _groq_chat_model()

    # Define the model's role and expected response behaviour.
    detailed_system_message = SystemMessage(
        content="""
    You are a senior Python developer with expertise
    in web frameworks.

    Instructions:
    1. Always provide practical code examples.
    2. Explain your code step by step.
    3. Follow Python best practices.
    4. Be concise but thorough.
    5. Use clear and understandable language.
    """
    )

    # Provide the actual user question.
    rest_api_human_message = HumanMessage(
        content="How do I create a REST API?"
    )

    # Combine system instructions and user input.
    detailed_instruction_messages = [
        detailed_system_message,
        rest_api_human_message
    ]

    # The model uses the system instructions to formulate
    # its response to the user's question.
    detailed_system_response = detailed_system_model.invoke(
        detailed_instruction_messages
    )

    print(
        "[Example 4 | detailed system] Reply text:",
        detailed_system_response.content
    )


# ============================================================
# EXAMPLE 5: MESSAGE METADATA
# ============================================================

def example_5_message_metadata():
    _require_groq_key()

    print("\n========== EXAMPLE 5: MESSAGE METADATA ==========")

    metadata_model = _groq_chat_model()

    # Create a HumanMessage with additional metadata.
    metadata_human_message = HumanMessage(
        content="Hello!",

        # Optional name identifying the message sender.
        name="alice",

        # Optional unique identifier for tracing or tracking.
        id="msg_123"
    )

    # Send the message to the model.
    metadata_response = metadata_model.invoke([metadata_human_message])

    print(
        "[Example 5 | message metadata] Reply text:",
        metadata_response.content
    )

    # Inspect message information.
    print("[Example 5 | message metadata] Content:", metadata_human_message.content)
    print("[Example 5 | message metadata] Name:", metadata_human_message.name)
    print("[Example 5 | message metadata] ID:", metadata_human_message.id)
    print("[Example 5 | message metadata] Type:", metadata_human_message.type)


# ============================================================
# EXAMPLE 6: AI MESSAGE HISTORY
# ============================================================

def example_6_aimessage_history():
    _require_groq_key()

    print("\n========== EXAMPLE 6: AI MESSAGE HISTORY ==========")

    history_model = _groq_chat_model()

    # Create an AIMessage manually.
    #
    # Normally, the LLM generates AIMessage automatically.
    # Here, we create one manually to demonstrate how
    # conversation history can be constructed.

    history_ai_message = AIMessage(
        content="I'd be happy to help you with that question!"
    )

    # Construct a conversation history.
    history_messages = [
        SystemMessage(
            content="You are a helpful assistant."
        ),

        HumanMessage(
            content="Can you help me?"
        ),

        # Insert a previous assistant response.
        history_ai_message,

        # Add the user's next question.
        HumanMessage(
            content="Great! What's 2 + 2?"
        )
    ]

    # Send the complete conversation history to the model.
    #
    # The model can use previous messages as context
    # when generating its next response.

    history_response = history_model.invoke(history_messages)

    print(
        "[Example 6 | AIMessage history] Reply text:",
        history_response.content
    )


# ============================================================
# EXAMPLE 7: TOOL MESSAGE
# ============================================================

def example_7_tool_message():
    _require_groq_key()

    print("\n========== EXAMPLE 7: TOOL MESSAGE ==========")

    tool_history_model = _groq_chat_model()

    # STEP 1: Create an AIMessage containing a tool-call request.
    #
    # The model requests the get_weather tool and provides
    # the required location argument.

    tool_call_ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_weather",
                "args": {
                    "location": "San Francisco"
                },
                "id": "call_123"
            }
        ]
    )

    # STEP 2: Simulate the execution of the Python tool.
    #
    # In a real application, you would execute your
    # get_weather() function or call a weather API.

    weather_result = "Sunny, 72°F"

    # STEP 3: Create a ToolMessage containing the tool result.
    #
    # tool_call_id MUST match the ID of the original
    # AIMessage tool call.
    #
    # This allows LangChain and the model to associate
    # the tool result with the correct request.

    weather_tool_message = ToolMessage(
        content=weather_result,
        tool_call_id="call_123",
        name="get_weather"
    )

    # STEP 4: Construct the conversation history.
    tool_history_messages = [
        HumanMessage(
            content="What's the weather in San Francisco?"
        ),

        # The model's original tool-call request.
        tool_call_ai_message,

        # The result returned by the tool.
        weather_tool_message
    ]

    # STEP 5: Send the complete conversation back to the LLM.
    #
    # The model can now read the weather result and
    # formulate a natural-language response.

    tool_history_response = tool_history_model.invoke(tool_history_messages)

    print(
        "[Example 7 | ToolMessage] Final reply:",
        tool_history_response.content
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n============================================")
    print("LANGCHAIN V1 - MESSAGES PRACTICE")
    print("============================================")

    # Uncomment the examples you want to execute.

    # example_1_plain_string()

    # example_2_text_prompt()

    # example_3_system_and_human()

    # example_4_detailed_system()

    # example_5_message_metadata()

    # example_6_aimessage_history()

    # example_7_tool_message()

    print("\n============================================")
    print("EXECUTION COMPLETED")
    print("============================================")


# Python entry point.
# This ensures main() runs only when this file
# is executed directly, not when imported elsewhere.

if __name__ == "__main__":
    main()
