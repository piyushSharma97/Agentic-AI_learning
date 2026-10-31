
### Structured output
# Models can be requested to provide their response in a format matching a given schema. 
# This is useful for ensuring the output can be easily parsed and used in subsequent processing.
#  LangChain supports multiple schema types and methods for enforcing structured output.


# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

import os
from dotenv import load_dotenv

# Load API keys from the .env file.
load_dotenv()

# Validate the Groq API key.
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from .env")


# ============================================================
# 2. INITIALIZE THE CHAT MODEL
# ============================================================

from langchain.chat_models import init_chat_model

# Initialize a Groq-hosted model.
# Use a model ID currently available to your Groq account.
model = init_chat_model("groq:openai/gpt-oss-120b")

# Test normal model invocation.
response = model.invoke("Explain structured output in AI.")

print(f"Response Content=>{response.content}")

# ============================================================
# 3. PYDANTIC STRUCTURED OUTPUT
# ============================================================

from pydantic import BaseModel, Field


# Define the expected structure of a movie.
class Movie(BaseModel):
    """A movie with its basic information."""

    # Movie title must be a string.
    title: str = Field(
        description="The title of the movie."
    )

    # Release year must be an integer.
    year: int = Field(
        description="The year the movie was released."
    )

    # Director's name must be a string.
    director: str = Field(
        description="The director of the movie."
    )

    # Rating must be a float between 0 and 10.
    rating: float = Field(
        description="The movie's rating out of 10.",
        ge=0,
        le=10
    )


# ============================================================
# 4. CONVERT THE MODEL TO STRUCTURED OUTPUT
# ============================================================

# with_structured_output() instructs the LLM to return
# data matching the Movie schema.
#
# function_calling uses tool/function calling to obtain
# structured data. It is useful with providers such as Groq.

model_with_structure = model.with_structured_output(
    Movie,
    method="function_calling"
)

# Send a prompt to the model.
response = model_with_structure.invoke(
    "Provide details about the movie Inception."
)

# The response is a validated Pydantic Movie instance.
print(f"Structure Output=> {response}")

# Access individual fields using dot notation.
print("\nMovie Title:", response.title)
print("Release Year:", response.year)
print("Director:", response.director)
print("Rating:", response.rating)

# Convert the Pydantic object into a Python dictionary.
print("\nDictionary:", response.model_dump())

# Convert the Pydantic object into JSON.
print("\nJSON:", response.model_dump_json(indent=2))

model.invoke("Provide details about the moview Inception")


response=model_with_structure.invoke("Provide details about the moview Inception")
response
# ============================================================
# 5. STRUCTURED OUTPUT WITH RAW RESPONSE
# ============================================================

# include_raw=True returns both:
# 1. The original AIMessage.
# 2. The parsed structured output.
# 3. Any parsing error.

model_with_structure = model.with_structured_output(
    Movie,
    method="function_calling",
    include_raw=True
)

response = model_with_structure.invoke(
    "Provide details about the movie Inception."
)

# Inspect the complete result.
print(response)

# Extract the original AIMessage.
raw_response = response["raw"]

print("\nRAW RESPONSE:")
print(raw_response)

# Extract the parsed Pydantic object.
parsed_response = response["parsed"]

print("\nPARSED RESPONSE:")
print(parsed_response)

# Inspect parsing errors, if any.
parsing_error = response["parsing_error"]

print("\nPARSING ERROR:")
print(parsing_error)

# Access structured fields when parsing succeeds.
if parsed_response is not None:
    print("\nMOVIE TITLE:", parsed_response.title)
    print("MOVIE YEAR:", parsed_response.year)

# ============================================================
# 6. NESTED PYDANTIC STRUCTURES
# ============================================================

from pydantic import BaseModel, Field


# Define the structure of an individual actor.
class Actor(BaseModel):
    """Information about a movie actor."""

    name: str = Field(
        description="The actor's full name."
    )

    role: str = Field(
        description="The character played by the actor."
    )


# Define the complete movie structure.
class MovieDetails(BaseModel):
    """Detailed information about a movie."""

    title: str = Field(
        description="The title of the movie."
    )

    year: int = Field(
        description="The release year."
    )

    # A list containing multiple Actor objects.
    # Every item must follow the Actor schema.
    cast: list[Actor] = Field(
        description="List of actors and their roles."
    )

    # A list containing multiple genre strings.
    genres: list[str] = Field(
        description="Genres of the movie."
    )

    # Budget is optional and can be None.
    # The field itself defaults to None.
    budget: float | None = Field(
        default=None,
        description="Budget in millions of USD."
    )


# Convert the model into a structured-output model.
model_with_structure = model.with_structured_output(
    MovieDetails,
    method="function_calling"
)

# Request nested movie information.
response = model_with_structure.invoke(
    "Provide detailed information about the movie Inception, "
    "including its cast, roles, genres and production budget. "
    "If any information is unknown, use an appropriate empty "
    "or null value rather than inventing it."
)

print(response)

# Access top-level fields.
print("\nMOVIE:", response.title)
print("YEAR:", response.year)
print("GENRES:", response.genres)
print("BUDGET:", response.budget)

# Iterate through the nested Actor objects.
print("\nCAST DETAILS:")

for actor in response.cast:
    print("Actor:", actor.name)
    print("Role:", actor.role)
    print("------------------------")


# ============================================================
# 7. STRUCTURED OUTPUT USING TYPEDDICT
# ============================================================

from typing_extensions import TypedDict, Annotated


# Define the expected structure of a movie dictionary.
class MovieDict(TypedDict):
    """A movie with basic information."""

    # Annotated adds a description to each field.
    title: Annotated[
        str,
        "The title of the movie."
    ]

    year: Annotated[
        int,
        "The year the movie was released."
    ]

    director: Annotated[
        str,
        "The director of the movie."
    ]

    rating: Annotated[
        float,
        "The movie's rating out of 10."
    ]


# Tell the model to return a dictionary matching MovieDict.
model_with_typeddict = model.with_structured_output(
    MovieDict,
    method="function_calling"
)

response = model_with_typeddict.invoke(
    "Provide the details of the movie Avengers: Endgame."
)

# TypedDict output is a normal Python dictionary.
print(response)

# Access fields using dictionary notation.
print("\nTITLE:", response["title"])
print("YEAR:", response["year"])
print("DIRECTOR:", response["director"])
print("RATING:", response["rating"])


# ============================================================
# 9. STRUCTURED OUTPUT USING DATACLASS
# ============================================================

from dataclasses import dataclass
from langchain.chat_models import init_chat_model


# Define a dataclass for contact information.
@dataclass
class ContactInfo:
    """Contact information for a person."""

    name: str
    email: str
    phone: str


# Convert the dataclass into a structured-output model.
model_with_dataclass = model.with_structured_output(
    ContactInfo,
    method="function_calling"
)

# Ask the LLM to extract contact information.
response = model_with_dataclass.invoke(
    """
    Extract contact information from the following text:

    John Doe, john@example.com, (555) 123-4567
    """
)

# The response is a ContactInfo dataclass instance.
print(response)

# Access fields using dot notation.
print("\nNAME:", response.name)
print("EMAIL:", response.email)
print("PHONE:", response.phone)

# ============================================================
# 10. STRUCTURED OUTPUT USING PYDANTIC + CREATE_AGENT
# ============================================================

from pydantic import BaseModel, Field
from langchain.agents import create_agent, ToolStrategy


# Define the expected contact information schema.
class ContactInfo(BaseModel):
    """Contact information extracted from text."""

    name: str = Field(
        description="The person's full name."
    )

    email: str = Field(
        description="The person's email address."
    )

    phone: str = Field(
        description="The person's phone number."
    )


# Create an agent with structured output.
agent = create_agent(
    model=model,
    tools=[],

    # ToolStrategy explicitly uses tool calling
    # to generate the structured response.
    # This is useful when the provider does not offer
    # native structured-output support.
    response_format=ToolStrategy(ContactInfo)
)

# Invoke the agent with a user message.
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": """
            Extract contact information from:

            John Doe, john@example.com, (555) 123-4567
            """
        }
    ]
})

# Extract the structured response from the agent state.
structured_response = result["structured_response"]

print(structured_response)

# Access the Pydantic object using dot notation.
print("\nNAME:", structured_response.name)
print("EMAIL:", structured_response.email)
print("PHONE:", structured_response.phone)


# ============================================================
# 11. STRUCTURED OUTPUT USING TYPEDDICT + CREATE_AGENT
# ============================================================

from typing_extensions import TypedDict
from langchain.agents import create_agent, ToolStrategy


# Define contact information using TypedDict.
class ContactInfoDict(TypedDict):
    """Contact information for a person."""

    name: str
    email: str
    phone: str


# Create an agent with TypedDict as its response schema.
agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(ContactInfoDict)
)

# Ask the agent to extract structured contact information.
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": """
            Extract contact information from:

            John Doe, john@example.com, (555) 123-4567
            """
        }
    ]
})

# Extract the structured response.
response = result["structured_response"]

# TypedDict responses are ordinary Python dictionaries.
print(response)

# Access fields using dictionary notation.
print("\nNAME:", response["name"])
print("EMAIL:", response["email"])
print("PHONE:", response["phone"])


# ============================================================
# 12. STRUCTURED OUTPUT USING DATACLASS + CREATE_AGENT
# ============================================================

from dataclasses import dataclass
from langchain.agents import create_agent, ToolStrategy


# Define contact information using a Python dataclass.
@dataclass
class ContactInfoData:
    """Contact information for a person."""

    name: str
    email: str
    phone: str


# Create an agent using the dataclass schema.
agent = create_agent(
    model=model,
    tools=[],
    response_format=ToolStrategy(ContactInfoData)
)

# Invoke the agent.
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": """
            Extract contact information from:

            John Doe, john@example.com, (555) 123-4567
            """
        }
    ]
})

# Extract the structured response.
response = result["structured_response"]

print(response)

# Access dataclass fields using dot notation.
print("\nNAME:", response.name)
print("EMAIL:", response.email)
print("PHONE:", response.phone)