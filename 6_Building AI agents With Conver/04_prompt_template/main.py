"""Lesson 4: build a prompt from reusable instructions and messages."""

import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not api_key:
    raise RuntimeError("Add GROQ_API_KEY to your .env file.")

model = ChatGroq(model=model_name, groq_api_key=api_key)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a patient tutor. Explain things in simple language."),
    MessagesPlaceholder(variable_name="messages"),
])

chain = prompt | model

response = chain.invoke({
    "messages": [
        HumanMessage(content="Explain the difference between a list and a tuple in Python.")
    ]
})

print(response.content)
