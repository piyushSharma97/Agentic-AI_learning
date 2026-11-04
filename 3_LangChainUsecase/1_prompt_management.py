"""
=============================================================================
USE CASE 1: PROMPT MANAGEMENT IN LANGCHAIN
=============================================================================
Prompt Management is the practice of systematically creating, templating,
validating, composing, and reusing prompts instead of hardcoding raw strings.

Key Concepts Covered:
1. PromptTemplate (Text-based prompts with input variables)
2. ChatPromptTemplate (Role-based message prompts: System, Human, AI)
3. MessagesPlaceholder (Dynamic message history injection)
4. Few-Shot Prompting (Teaching by example via FewShotChatMessagePromptTemplate)
5. Partial Prompt Formatting (Pre-filling fixed parameters early)
6. Prompt Composition (Combining multiple prompt pieces)
=============================================================================
"""

import os
from datetime import datetime
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# Import shared LLM config
from config import get_llm, safe_execute


# ---------------------------------------------------------------------------
# 1. Basic String PromptTemplate
# ---------------------------------------------------------------------------
def demo_basic_prompt_template():
    print("--- 1. String PromptTemplate ---")
    
    template = "Explain the concept of {topic} to a {audience_level} in {num_sentences} sentences."
    prompt = PromptTemplate.from_template(template)
    
    print(f"Input Variables required: {prompt.input_variables}")
    
    formatted_prompt = prompt.format(
        topic="Neural Networks",
        audience_level="10-year-old child",
        num_sentences=2
    )
    print(f"Formatted Output:\n{formatted_prompt}\n")


# ---------------------------------------------------------------------------
# 2. ChatPromptTemplate (System + Human + AI Roles)
# ---------------------------------------------------------------------------
def demo_chat_prompt_template():
    print("--- 2. ChatPromptTemplate (Multi-Role) ---")
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI tutor specialized in {subject}. Keep explanations clear and concise."),
        ("human", "What is the difference between {concept_a} and {concept_b}?")
    ])
    
    messages = chat_prompt.format_messages(
        subject="Computer Science",
        concept_a="Stack",
        concept_b="Queue"
    )
    for msg in messages:
        print(f"[{msg.type.upper()}]: {msg.content}")
    print()


# ---------------------------------------------------------------------------
# 3. MessagesPlaceholder (For Conversational History Injection)
# ---------------------------------------------------------------------------
def demo_messages_placeholder():
    print("--- 3. MessagesPlaceholder (Dynamic Chat History) ---")
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a customer support agent for CloudScale Inc."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_input}")
    ])
    
    # Simulated existing dialogue
    history = [
        HumanMessage(content="Hi, I cannot access my dashboard."),
        AIMessage(content="Hello! Have you tried clearing your browser cache and cookies?")
    ]
    
    formatted = chat_prompt.format_messages(
        chat_history=history,
        user_input="Yes, but it still shows 403 Forbidden."
    )
    
    for msg in formatted:
        print(f"[{msg.type.upper()}]: {msg.content}")
    print()


# ---------------------------------------------------------------------------
# 4. Few-Shot Chat Prompting (In-Context Learning)
# ---------------------------------------------------------------------------
def demo_few_shot_prompting():
    print("--- 4. Few-Shot Prompting (Teaching by Example) ---")
    
    # 1. Define few-shot examples
    examples = [
        {"input": "The battery lasts only 2 hours.", "sentiment": "Negative", "emotion": "Frustrated"},
        {"input": "Arrived earlier than expected, beautiful packaging!", "sentiment": "Positive", "emotion": "Delighted"},
        {"input": "It is a standard rectangular mouse pad.", "sentiment": "Neutral", "emotion": "Indifferent"}
    ]
    
    # 2. Define the template for each individual example
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "Sentiment: {sentiment} | Emotion: {emotion}")
    ])
    
    # 3. Wrap into FewShotChatMessagePromptTemplate
    few_shot_template = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )
    
    # 4. Final Chat Prompt combining System + Few-Shot + New Query
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an emotion and sentiment classifier. Follow the example format strictly."),
        few_shot_template,
        ("human", "{input}")
    ])
    
    formatted = final_prompt.format_messages(input="The software crashed while I was saving my report.")
    for msg in formatted:
        print(f"[{msg.type.upper()}]: {msg.content}")
    print()


# ---------------------------------------------------------------------------
# 5. Partial Prompt Formatting (Binding Static / Computed Values)
# ---------------------------------------------------------------------------
def demo_partial_formatting():
    print("--- 5. Partial Prompt Formatting ---")
    
    prompt = PromptTemplate(
        template="Today's date is {current_date}. Answer the user question: {question}",
        input_variables=["question"],
        partial_variables={"current_date": lambda: datetime.now().strftime("%Y-%m-%d")}
    )
    
    formatted = prompt.format(question="What is today's schedule?")
    print(f"Formatted Output:\n{formatted}\n")


# ---------------------------------------------------------------------------
# 6. Live Execution with LLM
# ---------------------------------------------------------------------------
def demo_live_llm_execution():
    print("--- 6. Live End-to-End LLM Prompt Pipeline ---")
    llm = get_llm()
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert software architect. Give a concise 2-sentence summary."),
        ("human", "What are the core advantages of microservices architecture?")
    ])
    
    # LCEL pipeline: Prompt -> LLM -> Output Parser
    chain = chat_prompt | llm | StrOutputParser()
    result = chain.invoke({})
    print(f"LLM Response:\n{result}\n")


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    safe_execute("Prompt Management Demonstrations", lambda: [
        demo_basic_prompt_template(),
        demo_chat_prompt_template(),
        demo_messages_placeholder(),
        demo_few_shot_prompting(),
        demo_partial_formatting(),
        demo_live_llm_execution(),
    ])


if __name__ == "__main__":
    main()

