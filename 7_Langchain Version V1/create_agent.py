from langchain.agents import create_agent
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[add],
    system_prompt="You are a helpful mathematical assistant."
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is 25 + 35?"}
    ]
})

print(response["messages"][-1].content)