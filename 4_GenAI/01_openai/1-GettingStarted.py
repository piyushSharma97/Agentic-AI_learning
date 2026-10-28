# os provides access to process environment variables used for API configuration.
import os
# python-dotenv loads local environment settings, such as an OpenAI key, from a .env file.
from dotenv import load_dotenv
# ChatOpenAI connects LangChain to an OpenAI chat model.
from langchain_openai import ChatOpenAI
# ChatPromptTemplate constructs reusable system and user messages with input placeholders.
from langchain_core.prompts import ChatPromptTemplate
# StrOutputParser converts a model message into plain text.
from langchain_core.output_parsers import StrOutputParser
# Read local settings before the OpenAI client is initialized.
load_dotenv()



# ChatOpenAI is a LangChain class that connects your Python application to an OpenAI chat model.

# Create the chat model client; credentials are read from the configured environment.
llm=ChatOpenAI(model="gpt-4o")
print("LLM=>",llm)

## Input and get response form LLM
# invoke sends a prompt directly to the model and returns a message object.
result=llm.invoke("What is generative AI?")

print("result content =>",result.content)
print("type of",type(result))

# Create a reusable prompt
# A prompt template separates reusable instructions from the question supplied for each call.
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert AI engineer. "
        "Explain technical concepts clearly, "
        "with practical examples."
    ),
    ("human", "{input}"),
])
prompt_value = prompt.invoke({
    "input": "Can you tell me about LangSmith?"
})

print("prompt_value=>",prompt_value)

## chain 
#The pipe operator connects components into a sequence. The output of one component becomes the input of the next.
# This chain formats messages and passes them to the chat model.
chain=prompt|llm
"""
formatted_prompt = prompt.invoke({"input": "What is LangSmith?"})
response = llm.invoke(formatted_prompt)
"""
response=chain.invoke({"input":"Can you tell me about HuggingFace?"})
print("chain=>",response)


print("Type of response=>",type(response))


## stroutput Parser
# Convert the model's AIMessage into plain text

# The parser changes an AIMessage into its plain-text content for simpler display and downstream use.
output_parser=StrOutputParser()
chain=prompt|llm|output_parser
print("Chain=>",chain)
response_chain=chain.invoke({"input":"Can you tell me about Langsmith?"})
print("response_chain=>",response_chain)
print(type(response))  # <class 'str'>

# ============================================================
# REVISION NOTES — 1-GettingStarted.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This file demonstrates calling an OpenAI chat model directly and then using it with a reusable LangChain prompt.
# It finishes by adding a string output parser so the chain returns plain text instead of a model message object.
#
# KEY CONCEPTS USED:
# • Chat model: accepts messages or prompts and returns an AI response.
# • Prompt template: stores reusable instructions and fills placeholders with each question.
# • Chain: connects prompt, model, and parser so each component's output feeds the next component.
#
# HOW IT CONNECTS TO THE PROJECT:
# This is a foundational OpenAI and LangChain example that prepares for the project's retrieval-based applications.
#
# THINGS TO REMEMBER:
# • OpenAI credentials must be available to the client through environment configuration.
# • A model response is typically a message object; StrOutputParser extracts its text.
# • The pipe operator composes runnable LangChain components into a pipeline.
#
# COMMON MISTAKES TO AVOID:
# • Confusing prompt template invocation with model invocation; they produce different kinds of values.
# • Expecting a chain without an output parser to return a plain string.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Compare the response types before and after adding StrOutputParser to the prompt-model chain.
# ============================================================