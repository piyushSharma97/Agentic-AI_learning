"""
Build a Simple LLM Application with LCEL
(LangChain Expression Language)

What you learn here:
1. How to talk to an LLM using messages
2. How to parse LLM output into clean strings
3. How to chain components using LCEL pipe operator (|)
4. How to use prompt templates for reusable prompts
5. The full chain: prompt | model | parser
"""

import os
# os lets us read environment variables (like API keys) from the system

from dotenv import load_dotenv
# dotenv reads your .env file and loads variables into the environment
# So you don't hardcode secrets like API keys in your code

from langchain_groq import ChatGroq
# ChatGroq connects to Groq's API — a fast LLM inference provider
# Groq runs models like Llama, Mixtral at very high speed
# Alternative to OpenAI — often faster and cheaper

from langchain_core.messages import HumanMessage, SystemMessage
# These are message types LangChain uses to structure conversations
# SystemMessage = instructions to the AI (like a system prompt)
# HumanMessage = what the user says
# Think of it like: System sets the rules, Human asks the question

from langchain_core.output_parsers import StrOutputParser
# LLMs return complex objects with metadata, tokens, etc.
# StrOutputParser extracts just the text content you actually want
# Input: AIMessage object → Output: plain string

from langchain_core.prompts import ChatPromptTemplate
# A template for creating reusable prompts with variables
# Instead of hardcoding "translate to French" every time
# You create a template with {language} and {text} placeholders


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()
# Reads .env file and loads GROQ_API_KEY into environment
# Your .env file should have: GROQ_API_KEY=your-key-here


# --------------------------------------------------
# 2. Initialize the LLM model
# --------------------------------------------------

model = ChatGroq(
    model="openai/gpt-oss-120b",
    # The specific model to use
    # Groq hosts many open source models
    
    temperature=0,
    # Controls randomness of responses
    # 0 = deterministic (same input always gives same output)
    # 1 = creative and varied
    # For translation: always use 0 — you want consistent results
)


# --------------------------------------------------
# 3. Direct message approach (simplest way)
# --------------------------------------------------

messages = [
    SystemMessage(
        content="Translate the following from English to French."
        # This tells the AI its role and task
        # Always comes first in the messages list
    ),
    HumanMessage(
        content="Hello, how are you? I am Piyush Sharma a frontend Dev learning AI"
        # This is the actual text to translate
    ),
]

result = model.invoke(messages)
# .invoke() sends the messages to the LLM and waits for response
# result is an AIMessage object — contains content + metadata
print(f"result=>", result.content)
# result.content extracts just the text from the AIMessage object


# --------------------------------------------------
# 4. Output parser — cleaner way to get text
# --------------------------------------------------

parser = StrOutputParser()
# Creates a parser that extracts string content from LLM responses

parsed_result = parser.invoke(result)
# Takes the AIMessage object from step 3
# Returns just the plain text string — no metadata, no AIMessage wrapper
print(f"parsed_result => {parsed_result}")
# Same content as result.content but done through the parser pipeline


# --------------------------------------------------
# 5. LCEL Chain — model | parser
# THE MOST IMPORTANT CONCEPT IN THIS FILE
# --------------------------------------------------

chain = model | parser
# The pipe operator | is LCEL (LangChain Expression Language)
# It chains components together like a pipeline
# Data flows left to right:
# messages → model (gets AI response) → parser (extracts text)
#
# This is equivalent to:
# result = model.invoke(messages)
# final = parser.invoke(result)
# But written in one clean line

response = chain.invoke(messages)
# Now you invoke the whole chain at once
# LangChain handles passing output of each step to the next
print(f"response=>{response}")


# --------------------------------------------------
# 6. Prompt Templates — reusable prompts with variables
# --------------------------------------------------

generic_template = (
    "Translate the following text into {language}. "
    "Return only the translated text."
    # {language} is a placeholder — gets filled in later
    # This template works for ANY language, not just French
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generic_template),
        # System message uses the template with {language} variable
        
        ("human", "{text}"),
        # Human message uses {text} variable — the content to translate
    ]
)
# ChatPromptTemplate is a factory that creates properly formatted
# messages when you give it the variable values

prompt_result = prompt.invoke(
    {
        "language": "French",  # fills {language} in the template
        "text": "Hello",       # fills {text} in the human message
    }
)
# Returns a list of properly formatted messages
# Ready to be passed to the model

print(prompt_result.to_messages())
# Shows the final messages that will be sent to the LLM
# Useful for debugging — see exactly what the AI receives


# --------------------------------------------------
# 7. Full LCEL Chain — prompt | model | parser
# THE COMPLETE PATTERN YOU WILL USE EVERYWHERE
# --------------------------------------------------

translation_chain = prompt | model | parser
# The complete pipeline:
# 1. prompt: takes {language} and {text} → creates formatted messages
# 2. model: takes messages → returns AIMessage with translation
# 3. parser: takes AIMessage → returns clean string
#
# This is the standard LangChain pattern for production applications
# You will see prompt | model | parser in almost every LangChain project

response = translation_chain.invoke(
    {
        "language": "French",
        "text": "Hello, how are you?",
        # Just provide the variables — the chain handles everything else
    }
)

print("Translation:", response)
# Final clean translated text — no AIMessage, no metadata, just the answer


# ============================================================
# REVISION NOTES — langchain_lcel_translation.py
# ============================================================
#
# WHAT THIS FILE DOES:
# Demonstrates the core LangChain pattern for building LLM applications.
# Builds a translation app that converts English text to any language
# using Groq's fast LLM inference, output parsers, and prompt templates.
#
# KEY CONCEPTS USED:
#
# • LCEL (LangChain Expression Language):
#   The pipe operator | chains components together.
#   prompt | model | parser means data flows through each step automatically.
#   This is the foundation of EVERY LangChain application you will build.
#
# • Messages (SystemMessage + HumanMessage):
#   How LangChain structures conversations with LLMs.
#   System = instructions/role for the AI.
#   Human = the actual user input.
#   Always in this order: System first, Human second.
#
# • StrOutputParser:
#   LLMs return AIMessage objects with lots of metadata.
#   StrOutputParser strips everything and gives you just the text.
#   Always use this at the end of your chain.
#
# • ChatPromptTemplate:
#   Makes prompts reusable with {variable} placeholders.
#   Instead of hardcoding "translate to French" you write {language}.
#   Now the same chain works for Spanish, Hindi, German — any language.
#
# • Temperature=0:
#   For tasks needing consistency (translation, extraction, classification)
#   always use temperature=0.
#   For creative tasks (writing, brainstorming) use 0.7-1.0.
#
# HOW IT CONNECTS TO THE PROJECT:
# This is a standalone learning file demonstrating LCEL basics.
# The prompt | model | parser pattern you learned here is used in
# EVERY subsequent project — RAG chains, agent chains, and tools all
# follow this same pattern. Master this and everything else becomes easier.
#
# THINGS TO REMEMBER:
# • | is not "or" here — it means "pipe output to next component"
# • .invoke() runs the chain synchronously (waits for response)
# • .stream() runs it with streaming (tokens appear one by one)
# • .batch() runs multiple inputs at once
# • ChatPromptTemplate variables must match exactly — {language} not {Language}
#
# COMMON MISTAKES TO AVOID:
# • Forgetting StrOutputParser at end — you get AIMessage not string
# • Using wrong variable name in template — causes KeyError
# • Setting temperature too high for factual tasks — inconsistent results
# • Not loading .env before initialising model — API key not found error
# • Calling result.content when using parser — parser already extracts it
#
# THE LCEL PATTERN YOU WILL USE FOREVER:
# chain = prompt | model | parser
# response = chain.invoke({"variable": "value"})
# This two-line pattern builds 90% of LangChain applications.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at step 7 — translation_chain = prompt | model | parser
# That one line is the entire concept. Everything else is just setup.
# ============================================================