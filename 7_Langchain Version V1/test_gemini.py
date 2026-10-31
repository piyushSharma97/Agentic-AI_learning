# ============================================================
# GEMINI SDK (NO LANGCHAIN)
# ============================================================
#
# This file calls the Google GenAI SDK directly.
# genai.Client is the SDK client. generate_content() sends
# one prompt and returns the model text.
#
# Uncomment the example you want inside main().
# ============================================================

import os

from dotenv import load_dotenv
from google import genai

# Load API keys from the .env file.
# The Google key is checked inside the example, not at import time.
load_dotenv()


def example_1_gemini_sdk():
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY is missing from .env")

    print("\n========== EXAMPLE 1: GEMINI SDK ==========")

    # Client authenticates with the Google API key.
    gemini_client = genai.Client(
        api_key=google_api_key
    )

    # generate_content() sends the prompt to the named Gemini model.
    print("[Example 1 | Gemini SDK] Sending request...", flush=True)

    try:
        gemini_sdk_response = gemini_client.models.generate_content(
            model="gemini-3.8-flash",
            contents="Why do parrots talk?"
        )

        print(
            "[Example 1 | Gemini SDK] Reply text:",
            gemini_sdk_response.text
        )

    except Exception as gemini_sdk_error:
        print(
            "[Example 1 | Gemini SDK] Error:",
            type(gemini_sdk_error).__name__,
            str(gemini_sdk_error)
        )


def main():

    print("\n============================================")
    print("GEMINI SDK PRACTICE")
    print("============================================")

    # Uncomment the examples you want to execute.

    # example_1_gemini_sdk()

    print("\n============================================")
    print("EXECUTION COMPLETED")
    print("============================================")


# Python entry point.
# This ensures main() runs only when this file
# is executed directly, not when imported elsewhere.

if __name__ == "__main__":
    main()
