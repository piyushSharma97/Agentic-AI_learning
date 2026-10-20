import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()



# ChatOpenAI is a LangChain class that connects your Python application to an OpenAI chat model.

llm=ChatOpenAI(model="gpt-4o")
print(llm)

## Input and get response form LLM
result=llm.invoke("What is generative AI?")

print(result.content)
print(type(result))

# Create a reusable prompt
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
chain=prompt|llm
"""
formatted_prompt = prompt.invoke({"input": "What is LangSmith?"})
response = llm.invoke(formatted_prompt)
"""
response=chain.invoke({"input":"Can you tell me about HuggingFace?"})
print("chain=>",response)


type(response)


## stroutput Parser
# Convert the model's AIMessage into plain text

output_parser=StrOutputParser()
chain=prompt|llm|output_parser

response=chain.invoke({"input":"Can you tell me about Langsmith?"})
print(response)
print(type(response))  # <class 'str'>