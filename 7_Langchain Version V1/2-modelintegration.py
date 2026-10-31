# ============================================================
# LANGCHAIN V1: MODEL INTEGRATION, STREAMING AND BATCH
# ============================================================
#
# Examples covered:
#
# 1. Installed LangChain version
# 2. OpenAI through init_chat_model()
# 3. OpenAI through ChatOpenAI
#    Gemini through ChatGoogleGenerativeAI (example_gemini)
# 4. Groq through ChatGroq
# 5. Streaming
# 6. Batch
# 7. Batch with max_concurrency
#
# Each example builds its own model client.
# Uncomment the example you want inside main().
# example_gemini stays commented because the model name may fail.
# ============================================================


# ------------------------------------------------------------
# 1. ENVIRONMENT CONFIGURATION
# ------------------------------------------------------------

import os

from dotenv import load_dotenv

# Load environment variables from the .env file.
# This allows us to access API keys without hardcoding them.
# The key check lives inside the example that needs that provider,
# so uncommenting one example does not require every provider key.
load_dotenv()


def _require_api_key(env_name):
    """Raise only when the example that needs this key is running."""
    if not os.getenv(env_name):
        raise ValueError(f"{env_name} is missing from .env")


# ------------------------------------------------------------
# EXAMPLE 1: LANGCHAIN VERSION
# ------------------------------------------------------------

import langchain


def example_1_langchain_version():
    print("\n========== EXAMPLE 1: LANGCHAIN VERSION ==========")

    # Display the installed LangChain version.
    print("[Example 1 | LangChain version] Installed version:", langchain.__version__)


# ------------------------------------------------------------
# EXAMPLE 2: OPENAI MODEL INTEGRATION USING init_chat_model()
# ------------------------------------------------------------

# init_chat_model() is a unified factory function.
# It initializes a chat model using a provider:model identifier.
# This makes switching between model providers easier.

from langchain.chat_models import init_chat_model


def example_2_init_chat_model():
    _require_api_key("OPENAI_API_KEY")

    print("\n========== EXAMPLE 2: init_chat_model ==========")

    openai_init_model = init_chat_model("openai:gpt-4.1")

    # invoke() sends a single prompt to the model.
    # It waits until the complete response is generated.
    print("[Example 2 | init_chat_model] Starting request...", flush=True)
    openai_init_response = openai_init_model.invoke("Hello, how are you?")

    # invoke() returns an AIMessage object.
    # .content extracts the generated text.
    print("[Example 2 | init_chat_model] Reply text:", openai_init_response.content)
    print("[Example 2 | init_chat_model] Completed.", flush=True)


# ------------------------------------------------------------
# EXAMPLE 3: OPENAI INTEGRATION USING ChatOpenAI
# ------------------------------------------------------------

# ChatOpenAI is the provider-specific integration for OpenAI.
# It gives us direct access to OpenAI model configuration.

from langchain_openai import ChatOpenAI


def example_3_chat_openai():
    _require_api_key("OPENAI_API_KEY")

    print("\n========== EXAMPLE 3: ChatOpenAI ==========")

    print("[Example 3 | ChatOpenAI] Starting request...", flush=True)
    chat_openai_model = ChatOpenAI(
        model="gpt-4.1",
        temperature=0.7
    )

    # Send a prompt to the OpenAI model.
    chat_openai_response = chat_openai_model.invoke(
        "Hello, how are you?"
    )

    print("[Example 3 | ChatOpenAI] Completed.", flush=True)
    print("[Example 3 | ChatOpenAI] Reply text:", chat_openai_response.content)


# ------------------------------------------------------------
# EXAMPLE: GOOGLE GEMINI MODEL INTEGRATION
# ------------------------------------------------------------

# Install:
# pip install -U langchain-google-genai

# ChatGoogleGenerativeAI provides Google Gemini integration.
# The model string below may not be available on every account,
# which is why example_gemini() is left commented in main().

from langchain_google_genai import ChatGoogleGenerativeAI


def example_gemini():
    _require_api_key("GOOGLE_API_KEY")

    print("\n========== EXAMPLE: GEMINI ==========")

    print("[Example Gemini | ChatGoogleGenerativeAI] Starting request...", flush=True)
    gemini_model = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash"
    )

    # Send a prompt to Gemini.
    gemini_response = gemini_model.invoke(
        "Why do parrots talk?"
    )

    print("[Example Gemini | ChatGoogleGenerativeAI] Completed.", flush=True)
    print("[Example Gemini | ChatGoogleGenerativeAI] Reply text:", gemini_response.content)


# ------------------------------------------------------------
# EXAMPLE 4: GROQ MODEL INTEGRATION
# ------------------------------------------------------------

# Install:
# pip install -U langchain-groq

# ChatGroq connects LangChain to models served by Groq.

from langchain_groq import ChatGroq


def example_4_groq():
    _require_api_key("GROQ_API_KEY")

    print("\n========== EXAMPLE 4: GROQ ==========")

    groq_model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    # Send a prompt to the Groq-hosted model.
    groq_response = groq_model.invoke(
        "Why do parrots talk?"
    )

    print("[Example 4 | ChatGroq] Reply text:", groq_response.content)


# ------------------------------------------------------------
# EXAMPLE 5: STREAMING
# ------------------------------------------------------------

def example_5_streaming():
    _require_api_key("OPENAI_API_KEY")

    print("\n========== EXAMPLE 5: STREAMING ==========")

    # Normal invoke():
    # The application waits for the complete response.
    #
    # stream():
    # Returns an iterator that produces response chunks
    # progressively as they become available.

    streaming_model = init_chat_model("openai:gpt-4.1")

    streaming_prompt = """
Write a 200-word paragraph explaining
Artificial Intelligence and its applications.
"""

    print("[Example 5 | streaming] Reply text:")

    # Iterate over the generated response chunks.
    for stream_chunk in streaming_model.stream(streaming_prompt):

        # stream_chunk.text extracts the textual content of the chunk.
        #
        # end="" prevents print() from adding a newline
        # after every chunk.
        #
        # flush=True immediately displays the output
        # instead of waiting for the output buffer.

        print(stream_chunk.text, end="", flush=True)

    print("\n[Example 5 | streaming] Completed.")


# ------------------------------------------------------------
# EXAMPLE 6: BATCH PROCESSING
# ------------------------------------------------------------

def example_6_batch():
    _require_api_key("OPENAI_API_KEY")

    print("\n========== EXAMPLE 6: BATCH ==========")

    # batch() accepts multiple independent prompts.
    #
    # Instead of invoking the model separately for every prompt,
    # we submit all prompts together.
    #
    # The default batch implementation can execute requests
    # concurrently, depending on the model integration.

    batch_model = init_chat_model("openai:gpt-4.1")

    batch_questions = [
        "Why do parrots have colorful feathers?",
        "How do airplanes fly?",
        "What is quantum computing?"
    ]

    batch_responses = batch_model.batch(batch_questions)

    # batch() returns a list of AIMessage objects.
    # Each response corresponds to its input prompt.
    print("[Example 6 | batch] Reply list start:")
    for index, batch_response in enumerate(batch_responses, start=1):

        print(f"[Example 6 | batch] Question {index}:", batch_questions[index - 1])

        # Extract the generated response text.
        print(f"[Example 6 | batch] Answer {index}:", batch_response.content)

    print("[Example 6 | batch] Reply list end.")


# ------------------------------------------------------------
# EXAMPLE 7: BATCH PROCESSING WITH MAX_CONCURRENCY
# ------------------------------------------------------------

def example_7_batch_max_concurrency():
    _require_api_key("OPENAI_API_KEY")

    print("\n========== EXAMPLE 7: BATCH MAX CONCURRENCY ==========")

    # max_concurrency controls the maximum number of
    # concurrent executions.
    #
    # For example:
    # If there are 10 prompts and max_concurrency = 5,
    # LangChain will execute up to 5 requests concurrently.
    #
    # The remaining requests execute as capacity becomes available.

    batch_concurrent_model = init_chat_model("openai:gpt-4.1")

    batch_concurrent_questions = [
        "Why do parrots have colorful feathers?",
        "How do airplanes fly?",
        "What is quantum computing?"
    ]

    batch_concurrent_responses = batch_concurrent_model.batch(
        batch_concurrent_questions,
        config={
            "max_concurrency": 5
        }
    )

    print("[Example 7 | batch max_concurrency] Reply list:")

    for batch_concurrent_response in batch_concurrent_responses:
        print(
            "[Example 7 | batch max_concurrency] Reply text:",
            batch_concurrent_response.content
        )
        print("-" * 50)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("\n============================================")
    print("LANGCHAIN V1 - MODEL INTEGRATION PRACTICE")
    print("============================================")

    # Uncomment the examples you want to execute.

    # example_1_langchain_version()

    # example_2_init_chat_model()

    # example_3_chat_openai()

    # example_gemini()

    # example_4_groq()

    # example_5_streaming()

    # example_6_batch()

    # example_7_batch_max_concurrency()

    print("\n============================================")
    print("EXECUTION COMPLETED")
    print("============================================")


# Python entry point.
# This ensures main() runs only when this file
# is executed directly, not when imported elsewhere.

if __name__ == "__main__":
    main()


# ============================================================
# END OF LANGCHAIN MODEL INTEGRATION
# ============================================================
