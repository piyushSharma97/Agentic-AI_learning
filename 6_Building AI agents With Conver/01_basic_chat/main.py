"""Lesson 1: send one message to a chat model."""

from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not api_key:
    raise RuntimeError("Add GROQ_API_KEY to your .env file.")

# Create the model client. This does not send a question yet.
model = ChatGroq(model=model_name, groq_api_key=api_key)

# Send one user message to the model.
response = model.invoke([
    HumanMessage(content="Explain what an LLM is in two simple sentences.")
])

# AIMessage.content contains the generated text.
print(response.content)
