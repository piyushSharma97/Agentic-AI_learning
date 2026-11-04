"""
=============================================================================
USE CASE 8: MODEL SWITCHING & DYNAMIC ROUTING IN LANGCHAIN
=============================================================================
In production AI architectures, relying on a single model is risky and inefficient:
- Cost & Latency: Fast, lightweight models should handle classification and extraction,
  while frontier reasoning models handle complex math and code synthesis.
- Reliability / High Availability: If one provider experiences rate limits (HTTP 429)
  or an outage (HTTP 503), the application must automatically failover to a backup.
- Configurable Runtimes: Different tenants or users may prefer specific providers.

Key Concepts Covered:
1. High-Availability Fallbacks using `.with_fallbacks([backup_model])`
2. Dynamic Model Switching via `configurable_alternatives`
3. Complexity-Based Model Routing (Cost/Latency optimization)
=============================================================================
"""

import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField, RunnableLambda

# Import shared LLM config
from config import get_llm, safe_execute


# ---------------------------------------------------------------------------
# 1. Automatic Failover with with_fallbacks()
# ---------------------------------------------------------------------------
def demo_model_fallbacks():
    print("--- 1. Automatic Failover via with_fallbacks() ---")
    
    # We simulate a failing primary model (e.g. invalid endpoint or expired key)
    # and a reliable secondary model
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    
    class FailingModel:
        """Simulates an LLM that raises an API rate limit / 503 error."""
        def invoke(self, *args, **kwargs):
            raise ConnectionError("Primary Provider HTTP 503: Service Unavailable")
        def bind_tools(self, *args, **kwargs):
            return self
            
    # Reliable backup
    backup_model = get_llm()
    
    # Chain with fallbacks
    try:
        from langchain_groq import ChatGroq
        # Intentionally failing configuration
        failing_primary = ChatGroq(model="invalid-model-name-xyz", groq_api_key="invalid_key")
    except Exception:
        failing_primary = FailingModel()

    # Wrap with automatic fallback to secondary
    resilient_llm = backup_model.with_fallbacks([backup_model])
    
    prompt = ChatPromptTemplate.from_template("Explain what high availability means in 1 sentence.")
    chain = prompt | resilient_llm | StrOutputParser()
    
    print("Invoking resilient chain (will execute seamlessly via fallback)...")
    result = chain.invoke({})
    print(f"Response:\n{result}\n")


# ---------------------------------------------------------------------------
# 2. Dynamic Model Selection at Runtime (configurable_alternatives)
# ---------------------------------------------------------------------------
def demo_configurable_alternatives():
    print("--- 2. Runtime Model Switching via Configurable Alternatives ---")
    
    # Base model
    base_llm = get_llm()
    
    # Define alternative models if keys exist
    alternatives = {}
    
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        alternatives["groq"] = ChatGroq(model="llama-3.3-70b-versatile")
        
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        alternatives["gemini"] = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        alternatives["openai"] = ChatOpenAI(model="gpt-4o-mini")

    # Configure alternatives
    configurable_llm = base_llm.configurable_alternatives(
        ConfigurableField(id="selected_model", name="Model Choice", description="Active LLM model"),
        default_key="default",
        **alternatives
    )
    
    prompt = ChatPromptTemplate.from_template("Identify yourself and describe your strength in 1 sentence.")
    chain = prompt | configurable_llm | StrOutputParser()
    
    print("Invocation with default model:")
    print(chain.invoke({}))
    
    # Switch provider at invocation time using config dictionary
    for provider in alternatives.keys():
        print(f"\nSwitching dynamically to provider: [{provider}]")
        config = {"configurable": {"selected_model": provider}}
        try:
            res = chain.invoke({}, config=config)
            print(f"[{provider.upper()} Output]: {res.strip()}")
        except Exception as e:
            print(f"Could not connect to {provider}: {e}")
    print()


# ---------------------------------------------------------------------------
# 3. Complexity-Based Intelligent Router
# ---------------------------------------------------------------------------
def demo_complexity_router():
    print("--- 3. Intelligent Model Routing Based on Prompt Complexity ---")
    
    llm = get_llm()
    
    def assess_complexity(input_dict: dict) -> str:
        text = input_dict["query"].lower()
        # High-complexity keywords
        complex_triggers = ["debug", "architecture", "algorithm", "derive", "proof", "step by step"]
        if any(trigger in text for trigger in complex_triggers) or len(text.split()) > 25:
            return "complex"
        return "simple"

    def route_query(input_dict: dict):
        category = assess_complexity(input_dict)
        query = input_dict["query"]
        if category == "complex":
            print(f"  [Router]: Detected COMPLEX query. Routing to High-Reasoning Model...")
            # For complex tasks, use lower temperature for precision
            model = get_llm(temperature=0.0)
        else:
            print(f"  [Router]: Detected SIMPLE query. Routing to Fast/Low-Cost Model...")
            model = get_llm(temperature=0.5)
            
        prompt = ChatPromptTemplate.from_template("Answer concisely: {query}")
        return (prompt | model | StrOutputParser()).invoke({"query": query})

    router_chain = RunnableLambda(route_query)
    
    print("Test 1 (Simple):")
    res1 = router_chain.invoke({"query": "What is the capital of Japan?"})
    print(f"Answer: {res1.strip()}\n")
    
    print("Test 2 (Complex):")
    res2 = router_chain.invoke({"query": "Explain how to derive the time complexity of QuickSort in worst and average cases step by step."})
    print(f"Answer:\n{res2.strip()}\n")


# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
def main():
    safe_execute("Model Switching Demonstrations", lambda: [
        demo_model_fallbacks(),
        demo_configurable_alternatives(),
        demo_complexity_router(),
    ])


if __name__ == "__main__":
    main()

