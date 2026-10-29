"""
Getting Started with Generative AI: OpenAI & LangChain Expression Language (LCEL)

What you will learn in this tutorial:
1. Connecting to OpenAI's latest chat models (gpt-4o, gpt-4o-mini).
2. Understanding the difference between raw model outputs (AIMessage) and plain text.
3. Building reusable, structured prompts with ChatPromptTemplate and system roles.
4. The LCEL Pipe Operator (|): Composing runnables into clean, declarative pipelines.
5. StrOutputParser: Stripping response metadata to extract clean string responses.
"""

import os
# python-dotenv loads OpenAI API credentials from a local .env file.
from dotenv import load_dotenv

# ChatOpenAI integrates LangChain with OpenAI's chat completions API.
from langchain_openai import ChatOpenAI

# ChatPromptTemplate builds reusable message templates with role definitions.
from langchain_core.prompts import ChatPromptTemplate

# StrOutputParser converts complex AIMessage objects into clean Python strings.
from langchain_core.output_parsers import StrOutputParser

# --------------------------------------------------
# 1. Environment Configuration
# --------------------------------------------------

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Warning: OPENAI_API_KEY is not set in environment or .env file.")

# --------------------------------------------------
# 2. Initializing the OpenAI Chat Model
# --------------------------------------------------

# ChatOpenAI wraps the OpenAI API:
# • model="gpt-4o": OpenAI's flagship multimodal model.
# • temperature: Defaults to 0.7 (controls creativity). Set to 0 for deterministic answers.
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
print("Step 1: Initialized ChatOpenAI client:")
print(llm)

# --------------------------------------------------
# 3. Direct Model Invocation (Simplest Way)
# --------------------------------------------------

# .invoke() sends a prompt to the model and waits for the complete response.
print("\nStep 2: Sending direct prompt to model...")
result = llm.invoke("What is generative AI?")

# The model returns an AIMessage object, NOT a raw string!
# AIMessage contains the generated text PLUS metadata (token usage, model name, finish reason).
print("\nType of result:", type(result))
print("Raw AIMessage content:\n", result.content[:150] + "...")

# --------------------------------------------------
# 4. Creating Reusable Prompts with ChatPromptTemplate
# --------------------------------------------------

# Why use ChatPromptTemplate instead of Python f-strings?
# 1. Role Enforcement: Clearly separates "system" rules from "human" input.
# 2. Reusability: Defines the prompt structure once and dynamically injects variables.
# 3. Security: Sanitizes inputs and prevents prompt injection attacks.
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert AI engineer. "
        "Explain technical concepts clearly, "
        "with practical examples."
    ),
    ("human", "{input}"),
])

# Inspecting what the formatted prompt looks like before sending to the model:
prompt_value = prompt.invoke({"input": "Can you tell me about LangSmith?"})
print("\nStep 3: Formatted PromptValue:")
print(prompt_value)

# --------------------------------------------------
# 5. Composing a Basic LCEL Chain (prompt | llm)
# --------------------------------------------------

# The pipe operator (|) is LangChain Expression Language (LCEL).
# It links components like an assembly line:
# Output of prompt (formatted messages) -> Input of llm (generates response)
chain = prompt | llm

# Invoking the chain automatically formats the prompt and calls the model in one clean step:
response = chain.invoke({"input": "Can you tell me about HuggingFace?"})
print("\nStep 4: Response from (prompt | llm) chain:")
print("Type of response:", type(response))  # Still an AIMessage object!
print("Content:\n", response.content[:150] + "...")

# --------------------------------------------------
# 6. Adding StrOutputParser (prompt | llm | output_parser)
# THE UNIVERSAL LANGCHAIN PRODUCTION PATTERN
# --------------------------------------------------

# StrOutputParser extracts just the clean string from the AIMessage object.
output_parser = StrOutputParser()

# The complete production pipeline:
# 1. prompt: formats dictionary variables into system/human messages.
# 2. llm: sends messages to OpenAI and returns AIMessage.
# 3. output_parser: parses AIMessage into clean str.
full_chain = prompt | llm | output_parser

print("\nStep 5: Invoking full chain (prompt | llm | output_parser)...")
final_response = full_chain.invoke({"input": "Can you tell me about LangSmith in 2 sentences?"})

print("Type of final_response:", type(final_response))  # <class 'str'>
print("Clean String Output:\n", final_response)

# ============================================================
# REVISION NOTES — 1-GettingStarted.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script is the foundational entry point for OpenAI application development with LangChain.
# It progresses from direct model invocation to structured prompt templates, LCEL pipeline chaining,
# and string output parsing.
#
# KEY CONCEPTS USED:
# • ChatOpenAI: LangChain class interfacing with OpenAI's chat completion endpoints.
# • AIMessage: Structured response object containing `.content` (text) and `.response_metadata` (tokens, finish reason).
# • ChatPromptTemplate: Template factory separating system instructions from dynamic user inputs.
# • LCEL Pipe Operator (|): Declarative syntax for connecting runnables into streamlined data pipelines.
# • StrOutputParser: Output transformer extracting the plain text payload from an AIMessage.
#
# THE CANONICAL LCEL PATTERN:
#   chain = prompt | model | parser
#   result = chain.invoke({"variable": "value"})
# This exact two-line pattern is used in 95% of all LangChain applications!
#
# THINGS TO REMEMBER:
# • ChatOpenAI requires an OPENAI_API_KEY environment variable.
# • prompt.invoke() takes a dictionary matching the placeholder names (e.g. `{"input": "..."}`).
# • Without an output parser, chain.invoke() returns an AIMessage, not a string.
#
# COMMON MISTAKES TO AVOID:
# • Hardcoding dynamic variables directly into string templates instead of using `{placeholder}` syntax.
# • Attempting to perform string operations on raw model responses without calling `.content` or using `StrOutputParser`.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at Step 6: `full_chain = prompt | llm | output_parser` and `final_response = full_chain.invoke({"input": "..."})`.
# ============================================================
