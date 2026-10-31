
# ============================================================
# LANGCHAIN V1: MODEL INTEGRATION, STREAMING AND BATCH
# ============================================================

# ------------------------------------------------------------
# 1. ENVIRONMENT CONFIGURATION
# ------------------------------------------------------------

import os
from dotenv import load_dotenv

# Load environment variables from the .env file.
# This allows us to access API keys without hardcoding them.
load_dotenv()

# Retrieve API keys from environment variables.
# load_dotenv() already makes them available to provider SDKs.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate API keys before making API requests.
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing from .env")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")


# ------------------------------------------------------------
# 2. LANGCHAIN VERSION
# ------------------------------------------------------------

import langchain

# Display the installed LangChain version.
print("LangChain Version:", langchain.__version__)


# ------------------------------------------------------------
# 3. OPENAI MODEL INTEGRATION USING init_chat_model()
# ------------------------------------------------------------

# init_chat_model() is a unified factory function.
# It initializes a chat model using a provider:model identifier.
# This makes switching between model providers easier.

from langchain.chat_models import init_chat_model

openai_model = init_chat_model("openai:gpt-4.1")

# invoke() sends a single prompt to the model.
# It waits until the complete response is generated.
print("\nStarting OpenAI init_chat_model...", flush=True)
response = openai_model.invoke("Hello, how are you?")

# invoke() returns an AIMessage object.
# .content extracts the generated text.

print(response.content)
print("OpenAI init_chat_model completed!", flush=True)

# ------------------------------------------------------------
# 4. OPENAI INTEGRATION USING ChatOpenAI
# ------------------------------------------------------------

# ChatOpenAI is the provider-specific integration for OpenAI.
# It gives us direct access to OpenAI model configuration.

from langchain_openai import ChatOpenAI
print("\nStarting ChatOpenAI...", flush=True)
openai_model_direct = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.7
)

# Send a prompt to the OpenAI model.
response = openai_model_direct.invoke(
    "Hello, how are you?"
)
print("ChatOpenAI completed!", flush=True)
print("\nCHAT OPENAI RESPONSE:")
print(response.content)


# ------------------------------------------------------------
# 5. GOOGLE GEMINI MODEL INTEGRATION
# ------------------------------------------------------------

# Install:
# pip install -U langchain-google-genai

# ChatGoogleGenerativeAI provides Google Gemini integration.

from langchain_google_genai import ChatGoogleGenerativeAI
print("\nStarting Gemini...", flush=True)
# gemini_model = ChatGoogleGenerativeAI(
#     model="gemini-3.8-flash"
# )

# Send a prompt to Gemini.
# response = gemini_model.invoke(
#     "Why do parrots talk?"
# )

# print("\nGEMINI RESPONSE:")
# print("Gemini completed!", flush=True)
# print(response.content)


# ------------------------------------------------------------
# 6. GROQ MODEL INTEGRATION
# ------------------------------------------------------------

# Install:
# pip install -U langchain-groq

# ChatGroq connects LangChain to models served by Groq.

from langchain_groq import ChatGroq

groq_model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# Send a prompt to the Groq-hosted model.
response = groq_model.invoke(
    "Why do parrots talk?"
)

print("\nGROQ RESPONSE:")
print(response.content)


# ------------------------------------------------------------
# 7. STREAMING
# ------------------------------------------------------------

# Normal invoke():
# The application waits for the complete response.
#
# stream():
# Returns an iterator that produces response chunks
# progressively as they become available.

prompt = """
Write a 200-word paragraph explaining
Artificial Intelligence and its applications.
"""

print("\nSTREAMING RESPONSE:\n")

# Iterate over the generated response chunks.
for chunk in openai_model.stream(prompt):

    # chunk.text extracts the textual content of the chunk.
    #
    # end="" prevents print() from adding a newline
    # after every chunk.
    #
    # flush=True immediately displays the output
    # instead of waiting for the output buffer.

    print(chunk.text, end="", flush=True)

print("\n\nStreaming completed!")


# ------------------------------------------------------------
# 8. BATCH PROCESSING
# ------------------------------------------------------------

# batch() accepts multiple independent prompts.
#
# Instead of invoking the model separately for every prompt,
# we submit all prompts together.
#
# The default batch implementation can execute requests
# concurrently, depending on the model integration.

questions = [
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]

responses = openai_model.batch(questions)

# batch() returns a list of AIMessage objects.
# Each response corresponds to its input prompt.
print("\n BATCH RESPONSE START:")
for index, response in enumerate(responses, start=1):

    print(f"\nQuestion {index}: {questions[index - 1]}")

    # Extract the generated response text.
    print("Answer:", response.content)

print("\n BATCH RESPONSE END:")
# ------------------------------------------------------------
# 9. BATCH PROCESSING WITH MAX_CONCURRENCY
# ------------------------------------------------------------

# max_concurrency controls the maximum number of
# concurrent executions.
#
# For example:
# If there are 10 prompts and max_concurrency = 5,
# LangChain will execute up to 5 requests concurrently.
#
# The remaining requests execute as capacity becomes available.

responses = openai_model.batch(
    questions,
    config={
        "max_concurrency": 5
    }
)

print("\nBATCH RESPONSE WITH MAX CONCURRENCY:\n")

for response in responses:
    print(response.content)
    print("-" * 50)


# ============================================================
# END OF LANGCHAIN MODEL INTEGRATION
# ============================================================
