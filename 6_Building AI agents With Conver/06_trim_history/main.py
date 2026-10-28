"""Lesson 6: keep a recent slice of a longer message history."""

import os
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_groq import ChatGroq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not api_key:
    raise RuntimeError("Add GROQ_API_KEY to your .env file.")

model = ChatGroq(model=model_name, groq_api_key=api_key)

messages = [
    SystemMessage(content="You are a concise assistant."),
    HumanMessage(content="My favorite ice cream is vanilla."),
    AIMessage(content="Vanilla is a classic choice."),
    HumanMessage(content="I also enjoy hiking."),
    AIMessage(content="Hiking is a great outdoor activity."),
    HumanMessage(content="What is 2 + 2?"),
    AIMessage(content="4."),
    HumanMessage(content="What is my favorite ice cream?"),
]

# Keep the system instruction outside the trimmed conversation.
system_message = messages[0]
conversation = messages[1:]

trimmed_conversation = trim_messages(
    conversation,
    max_tokens=350,
    strategy="last",
    token_counter=model,
    include_system=False,
    allow_partial=False,
    start_on="human",
)

print("Messages before trimming:", len(conversation))
print("Messages after trimming:", len(trimmed_conversation))

response = model.invoke([system_message, *trimmed_conversation])
print("\nAssistant:", response.content)
