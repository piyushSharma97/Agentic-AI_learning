from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

print("Sending request to Gemini...", flush=True)

try:
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Why do parrots talk?"
    )

    print(response.text)

except Exception as e:
    print(type(e).__name__, str(e))