"""
Lesson 5: Multi-Variable Prompts — Injecting Runtime Parameters

What you will learn in this lesson:
1. How to parameterize prompt templates with multiple dynamic variables.
2. How curly brace syntax `{variable}` defines runtime placeholders for system prompts and human queries.
3. How to instruct models to respond in specific target languages (e.g. Hindi, Spanish, French).
4. How LCEL chains validate and substitute all variables upon invocation.
"""

import os
from dotenv import load_dotenv

# ChatPromptTemplate allows creating structured templates with multiple dynamic placeholders.
from langchain_core.prompts import ChatPromptTemplate
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
# 2. Multi-Variable Prompt Template
# --------------------------------------------------

# Prompts can accept multiple variables simultaneously:
# - {language}: Dynamic target language inserted into the system instructions.
# - {question}: Dynamic text for the human query.
#
# This makes the prompt reusable for ANY language and ANY topic without changing code.
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful tutor. Answer in {language}. Keep the explanation beginner-friendly.",
    ),
    ("human", "{question}"),
])

# --------------------------------------------------
# 3. Composing & Invoking the Chain
# --------------------------------------------------

# Create the LCEL chain: prompt -> model
chain = prompt | model

# When invoking, provide values for all defined variables in a single dictionary.
# Here we request an explanation of Python functions in Hindi.
response = chain.invoke({
    "language": "Hindi",
    "question": "What is a Python function?",
})

print("Multi-lingual Tutor Response:\n")
print(response.content)

# ============================================================
# REVISION NOTES — 05_language_input/main.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script illustrates dynamic parameterization in LangChain prompt templates.
# By passing both `{language}` and `{question}` variables at runtime, it shows how a single prompt
# can be reused across different languages and queries while preserving consistent behavior instructions.
#
# KEY CONCEPTS USED:
# • Template Placeholders `{var}`: String interpolation syntax used by LangChain templates.
# • Multi-Variable Invocations: Supplying a dictionary containing all required variable bindings.
# • Cross-Lingual Prompting: Leveraging the multilingual pretraining of models by specifying response languages in system prompts.
#
# HOW IT CONNECTS TO THE PROJECT:
# This lesson demonstrates how chatbots can support user preferences (like changing languages on the fly).
# In Lesson 7, dynamic parameters ({language}) and conversation history ({history}) are combined together.
#
# THINGS TO REMEMBER:
# • Every variable name enclosed in `{}` in the template MUST be supplied in the input dictionary.
# • Variable names are case-sensitive: `{language}` is distinct from `{Language}`.
#
# COMMON MISTAKES TO AVOID:
# • Omitting one of the variables during invoke(), which raises a KeyError / InputValidationError.
# • Hardcoding target language or questions inside the system string instead of using variables.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Compare the `{language}` placeholder in the system tuple with `"language": "Hindi"` in `chain.invoke({...})`.
# ============================================================
