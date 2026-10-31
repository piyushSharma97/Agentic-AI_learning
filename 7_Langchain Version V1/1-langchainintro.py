### Langchain Version V1

### Agents

import langchain
from langchain.agents import create_agent
from dotenv import load_dotenv

print(langchain.__version__)


load_dotenv()




def get_weather(city:str)-> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny."


def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b

agent=create_agent(
    model="gpt-5",
    tools=[get_weather,calculate_sum],
    system_prompt="You are a helpful assistant."
)
agent

### run the agent
response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the weather like in New York?"
        }
    ]
})

print(response["messages"])
 
# Correct
response2 = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the weather in New York?"
        }
    ]
})

print(response2["messages"])



response3 = agent.invoke({
    "messages":[
        {
            "role":"user",
            "content": "What is the weather in Delhi and what is 25 + 35?"
        }
    ]
})

print(response3["messages"][-1].content)