"""
Lesson 4: Prompt Templates — Reusable System Instructions and MessagesPlaceholder

What you will learn in this lesson:
1. Why prompt templates are essential for controlling model persona and response format.
2. How to use system prompts to establish personality, constraints, and instructions.
3. How MessagesPlaceholder dynamically injects a variable list of conversation messages into the prompt.
4. How to compose prompt and model into an LCEL chain using the pipe operator (|).
"""

import os
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
# ChatPromptTemplate creates reusable message-based prompt templates.
# MessagesPlaceholder reserves an insertion point for a dynamically sized list of messages.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

# --------------------------------------------------
# 1. Environment & Model Initialization
# --------------------------------------------------

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not api_key:
    raise RuntimeError("Add GROQ_API_KEY to your .env file.")

model = ChatGroq(model=model_name, groq_api_key=api_key)

# --------------------------------------------------
# 2. Defining ChatPromptTemplate with MessagesPlaceholder
# --------------------------------------------------

# A ChatPromptTemplate combines static role definitions (system instructions)
# with dynamic slots for conversation history.
#
# ("system", ...) -> Fixes the persona/rules: here a patient, beginner-friendly tutor.
# MessagesPlaceholder(variable_name="messages") -> An elastic slot that receives a list of
# [HumanMessage, AIMessage, ...] and inserts them directly into the message array sent to the model.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a patient tutor. Explain things in simple language."),
    MessagesPlaceholder(variable_name="messages"),
])

# --------------------------------------------------
# 3. Composing the LCEL Chain
# --------------------------------------------------

# LangChain Expression Language (LCEL) uses the pipe operator (|) to link runnables:
# prompt (formats dictionary into messages) -> model (computes response)
chain = prompt | model

# --------------------------------------------------
# 4. Invoking the Chain
# --------------------------------------------------

# The input to the chain is a dictionary whose key matches the variable_name defined in MessagesPlaceholder.
response = chain.invoke({
    "messages": [
        HumanMessage(content="Explain the difference between a list and a tuple in Python.")
    ]
})

print("Tutor Response:\n")
print(response.content)

# ============================================================
# REVISION NOTES — 04_prompt_template/main.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script demonstrates combining system instructions with dynamic conversation messages.
# By defining a ChatPromptTemplate with a MessagesPlaceholder, it ensures that every message exchange
# is guided by the system prompt ("patient tutor") without manually concatenating strings.
#
# KEY CONCEPTS USED:
# • System Message: Guidance defining tone, domain boundaries, and persona for the AI model.
# • MessagesPlaceholder: LangChain construct that represents a variable-length list of chat messages.
# • LCEL (prompt | model): Declarative syntax piping structured prompts directly into the inference model.
#
# HOW IT CONNECTS TO THE PROJECT:
# This lesson adds steering and persona control to raw messages. In Lesson 7, MessagesPlaceholder is
# combined with RunnableWithMessageHistory and trimming to form the core production chatbot architecture.
#
# THINGS TO REMEMBER:
# • MessagesPlaceholder expects a list of Message objects (e.g. `[HumanMessage(...)]`), NOT a raw string.
# • The key name in the dictionary passed to invoke() must match `variable_name="messages"` exactly.
#
# COMMON MISTAKES TO AVOID:
# • Passing a string to MessagesPlaceholder instead of a list of messages.
# • Omitting system instructions, leaving the model without consistent tone or constraints.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at `MessagesPlaceholder(variable_name="messages")` and see how `chain.invoke({"messages": [...]})`
# fills that slot right after the system prompt.
# ============================================================
