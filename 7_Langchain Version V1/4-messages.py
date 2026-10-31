# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

import os
from dotenv import load_dotenv

# Load API keys from the .env file.
load_dotenv()

# Validate the Groq API key.
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from .env")


# ============================================================
# 2. INITIALIZE CHAT MODEL
# ============================================================

from langchain.chat_models import init_chat_model

# Initialize the Groq-hosted model.
# The provider:model format tells LangChain which integration
# and model to use.
model = init_chat_model("groq:openai/gpt-oss-120b")

# Send a simple text prompt to the model.
# A string is accepted as shorthand for a HumanMessage.
response = model.invoke(
    "Please explain Artificial Intelligence."
)

# Display the generated response.
print(response.content)

# ============================================================
# 3. TEXT PROMPTS
# ============================================================

# A plain string can be passed directly to invoke().
#
# LangChain treats this string as a user message.
#
# Text prompts are useful when:
# 1. You have a single, standalone question.
# 2. You don't need to maintain conversation history.
# 3. You want a simple interaction with an LLM.

response = model.invoke("What is LangChain?")

print(response.content)

# ============================================================
# 4. DIFFERENT MESSAGE TYPES
# ============================================================

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

# SystemMessage:
# Defines the behaviour, role and instructions of the LLM.
system_msg = SystemMessage(
    content="You are a poetry expert."
)

# HumanMessage:
# Represents the input provided by the user.
human_msg = HumanMessage(
    content="Write a poem about Artificial Intelligence."
)

# Create a conversation using multiple messages.
messages = [
    system_msg,
    human_msg
]

# Send the complete message list to the model.
response = model.invoke(messages)

# The model returns an AIMessage object.
print(response.content)

# ============================================================
# 5. SYSTEM MESSAGE WITH DETAILED INSTRUCTIONS
# ============================================================

from langchain_core.messages import SystemMessage, HumanMessage

# Define the model's role and expected response behaviour.
system_msg = SystemMessage(
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
human_msg = HumanMessage(
    content="How do I create a REST API?"
)

# Combine system instructions and user input.
messages = [
    system_msg,
    human_msg
]

# The model uses the system instructions to formulate
# its response to the user's question.
response = model.invoke(messages)

print(response.content)


# ============================================================
# 6. MESSAGE METADATA
# ============================================================

from langchain_core.messages import HumanMessage

# Create a HumanMessage with additional metadata.
human_msg = HumanMessage(
    content="Hello!",

    # Optional name identifying the message sender.
    name="alice",

    # Optional unique identifier for tracing or tracking.
    id="msg_123"
)

# Send the message to the model.
response = model.invoke([human_msg])

print(response.content)

# Inspect message information.
print("\nMESSAGE DETAILS:")
print("Content:", human_msg.content)
print("Name:", human_msg.name)
print("ID:", human_msg.id)
print("Type:", human_msg.type)


# ============================================================
# 7. AI MESSAGE
# ============================================================

from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    HumanMessage
)

# Create an AIMessage manually.
#
# Normally, the LLM generates AIMessage automatically.
# Here, we create one manually to demonstrate how
# conversation history can be constructed.

ai_msg = AIMessage(
    content="I'd be happy to help you with that question!"
)

# Construct a conversation history.
messages = [
    SystemMessage(
        content="You are a helpful assistant."
    ),

    HumanMessage(
        content="Can you help me?"
    ),

    # Insert a previous assistant response.
    ai_msg,

    # Add the user's next question.
    HumanMessage(
        content="Great! What's 2 + 2?"
    )
]

# Send the complete conversation history to the model.
#
# The model can use previous messages as context
# when generating its next response.

response = model.invoke(messages)

print(response.content)


# ============================================================
# 8. TOOL MESSAGE
# ============================================================

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage
)

# STEP 1: Create an AIMessage containing a tool-call request.
#
# The model requests the get_weather tool and provides
# the required location argument.

ai_message = AIMessage(
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

tool_message = ToolMessage(
    content=weather_result,
    tool_call_id="call_123",
    name="get_weather"
)

# STEP 4: Construct the conversation history.
messages = [
    HumanMessage(
        content="What's the weather in San Francisco?"
    ),

    # The model's original tool-call request.
    ai_message,

    # The result returned by the tool.
    tool_message
]

# STEP 5: Send the complete conversation back to the LLM.
#
# The model can now read the weather result and
# formulate a natural-language response.

response = model.invoke(messages)

print("\nFINAL RESPONSE:")
print(response.content)