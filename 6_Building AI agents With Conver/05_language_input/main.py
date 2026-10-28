"""Lesson 5: pass a language variable into a prompt."""

import os
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not api_key:
    raise RuntimeError("Add GROQ_API_KEY to your .env file.")

model = ChatGroq(model=model_name, groq_api_key=api_key)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful tutor. Answer in {language}. Keep the explanation beginner-friendly.",
    ),
    ("human", "{question}"),
])

chain = prompt | model

response = chain.invoke({
    "language": "Hindi",
    "question": "What is a Python function?",
})

print(response.content)
