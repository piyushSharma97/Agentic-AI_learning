"""Lesson 2: pass previous messages to the model manually."""

from dotenv import load_dotenv
import os

from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not api_key:
    raise RuntimeError("Add GROQ_API_KEY to your .env file.")

model = ChatGroq(model=model_name, groq_api_key=api_key)

conversation = [
    HumanMessage(content="Hi, my name is Krish."),
    AIMessage(content="Hi Krish! Nice to meet you."),
    HumanMessage(content="What is my name?"),
]

response = model.invoke(conversation)
print(response.content)
