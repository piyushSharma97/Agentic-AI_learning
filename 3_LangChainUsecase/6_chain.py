"""
=============================================================================
USE CASE 6: CHAINS & LCEL (LANGCHAIN EXPRESSION LANGUAGE)
=============================================================================
A Chain connects multiple components (prompts, models, parsers, functions)
into a reliable, deterministic execution pipeline.

In modern LangChain, chains are built using LCEL (LangChain Expression Language),
which provides:
- Pipe operator `|` composition
- Unified interface: `invoke()`, `batch()`, `stream()`, `ainvoke()`
- Seamless parallelization via `RunnableParallel`
- In-pipeline custom logic via `RunnableLambda`
- Input pass-through via `RunnablePassthrough`
- Conditional branching via `RunnableBranch`

Key Concepts Covered:
1. Sequential Chain: Prompt | Model | Parser
2. Parallel Chains: Executing multiple analyses concurrently
3. RunnablePassthrough: Preserving inputs alongside processed outputs
4. RunnableLambda: Integrating custom Python transformation functions
5. Conditional Branching: Routing inputs dynamically
6. Streaming tokens in real-time
=============================================================================
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
)

# Import shared LLM config
from config import get_llm, safe_execute


# ---------------------------------------------------------------------------
# 1. Simple Sequential Chain (Prompt | LLM | Parser)
# ---------------------------------------------------------------------------
def demo_sequential_chain():
    print("--- 1. Simple Sequential Chain ---")
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional copywriter. Write a catchy marketing slogan."),
        ("human", "Product: {product}\nTarget Audience: {target_audience}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({
        "product": "Ergonomic Standing Desk",
        "target_audience": "Remote Software Developers"
    })
    print(f"Generated Slogan:\n{result}\n")


# ---------------------------------------------------------------------------
# 2. Parallel Processing with RunnableParallel
# ---------------------------------------------------------------------------
def demo_parallel_chain():
    print("--- 2. Parallel Processing (RunnableParallel) ---")
    llm = get_llm()
    
    # Branch A: Generate a concise executive summary
    summary_prompt = ChatPromptTemplate.from_template(
        "Summarize this customer feedback in 1 sentence:\n\n{feedback}"
    )
    summary_chain = summary_prompt | llm | StrOutputParser()
    
    # Branch B: Assess sentiment and urgency
    sentiment_prompt = ChatPromptTemplate.from_template(
        "Classify sentiment (Positive/Neutral/Negative) and urgency (High/Medium/Low) for:\n\n{feedback}"
    )
    sentiment_chain = sentiment_prompt | llm | StrOutputParser()
    
    # Branch C: Recommend next action
    action_prompt = ChatPromptTemplate.from_template(
        "Suggest 1 clear operational next step to resolve or address:\n\n{feedback}"
    )
    action_chain = action_prompt | llm | StrOutputParser()
    
    # Combine all 3 into parallel execution
    analysis_chain = RunnableParallel({
        "original_text": RunnablePassthrough(),
        "summary": summary_chain,
        "sentiment_and_urgency": sentiment_chain,
        "recommended_action": action_chain
    })
    
    feedback_text = (
        "We have been experiencing payment gateway timeouts for the past 4 hours. "
        "Our customers cannot checkout and we are losing estimated thousands in sales! "
        "Need this resolved immediately!"
    )
    
    results = analysis_chain.invoke(feedback_text)
    
    print(f"Summary:\n{results['summary'].strip()}\n")
    print(f"Sentiment & Urgency:\n{results['sentiment_and_urgency'].strip()}\n")
    print(f"Recommended Action:\n{results['recommended_action'].strip()}\n")


# ---------------------------------------------------------------------------
# 3. Custom Transformations with RunnableLambda & RunnablePassthrough
# ---------------------------------------------------------------------------
def demo_lambda_and_passthrough():
    print("--- 3. Custom Python Logic with RunnableLambda ---")
    llm = get_llm()
    
    def count_words(text: str) -> dict:
        return {"word_count": len(text.split())}
        
    def clean_text(text: str) -> str:
        return text.strip().capitalize()
        
    pipeline = (
        RunnableLambda(clean_text)
        | RunnableParallel({
            "cleaned_input": RunnablePassthrough(),
            "stats": RunnableLambda(count_words),
            "critique": ChatPromptTemplate.from_template("Provide a 1-sentence critique: {text}") | llm | StrOutputParser()
        })
    )
    
    sample = "   artificial intelligence is transforming software engineering workflows.   "
    output = pipeline.invoke(sample)
    print(f"Cleaned Input: '{output['cleaned_input']}'")
    print(f"Stats: {output['stats']}")
    print(f"Critique: {output['critique'].strip()}\n")


# ---------------------------------------------------------------------------
# 4. Conditional Branching with RunnableBranch
# ---------------------------------------------------------------------------
def demo_conditional_branching():
    print("--- 4. Dynamic Conditional Branching (RunnableBranch) ---")
    llm = get_llm()
    
    # Chain for tech inquiries
    tech_chain = (
        ChatPromptTemplate.from_template("Provide technical troubleshooting steps for: {query}")
        | llm
        | StrOutputParser()
    )
    
    # Chain for billing inquiries
    billing_chain = (
        ChatPromptTemplate.from_template("Explain billing policies and refund protocols for: {query}")
        | llm
        | StrOutputParser()
    )
    
    # Default fallback chain
    general_chain = (
        ChatPromptTemplate.from_template("Provide general customer assistance for: {query}")
        | llm
        | StrOutputParser()
    )
    
    # Branching router based on keyword or classification
    router_chain = RunnableBranch(
        (lambda x: any(w in x["query"].lower() for w in ["bug", "error", "crash", "code", "server"]), tech_chain),
        (lambda x: any(w in x["query"].lower() for w in ["refund", "invoice", "charge", "card", "pricing"]), billing_chain),
        general_chain
    )
    
    test_queries = [
        "The server is throwing a 500 error when submitting the form.",
        "I was charged twice on my monthly invoice."
    ]
    
    for q in test_queries:
        print(f"Query: {q}")
        res = router_chain.invoke({"query": q})
        print(f"Routed Response:\n{res.strip()}\n" + "-" * 40)


# ---------------------------------------------------------------------------
# 5. Token Streaming with chain.stream()
# ---------------------------------------------------------------------------
def demo_streaming():
    print("--- 5. Real-Time Token Streaming (.stream) ---")
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_template("Give 3 bullet points on how to write clean code.")
    chain = prompt | llm | StrOutputParser()
    
    print("Streaming tokens directly: ", end="", flush=True)
    for chunk in chain.stream({}):
        print(chunk, end="", flush=True)
    print("\n")


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    safe_execute("Chains & LCEL Demonstrations", lambda: [
        demo_sequential_chain(),
        demo_parallel_chain(),
        demo_lambda_and_passthrough(),
        demo_conditional_branching(),
        demo_streaming(),
    ])


if __name__ == "__main__":
    main()

