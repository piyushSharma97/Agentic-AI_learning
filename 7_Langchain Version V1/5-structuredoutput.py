
### Structured output
# Models can be requested to provide their response in a format matching a given schema. 
# This is useful for ensuring the output can be easily parsed and used in subsequent processing.
#  LangChain supports multiple schema types and methods for enforcing structured output.


# ============================================================
# LANGCHAIN V1: STRUCTURED OUTPUT - COMPLETE PRACTICE FILE
# ============================================================
#
# Examples covered:
#
# 1. Pydantic - Product extraction
# 2. Nested Pydantic - Candidate / Resume extraction
# 3. TypedDict - Customer support classification
# 4. Dataclass - Employee information extraction
# 5. include_raw=True - Expense extraction and debugging
# 6. create_agent + ToolStrategy - Travel planning
#
# ============================================================


# ============================================================
# 1. IMPORTS AND ENVIRONMENT CONFIGURATION
# ============================================================

import os
from dotenv import load_dotenv

# Load environment variables from .env.
load_dotenv()

# Validate the Groq API key.
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from .env")


# ============================================================
# 2. INITIALIZE THE CHAT MODEL
# ============================================================

from langchain.chat_models import init_chat_model

# Initialize a Groq-hosted model.
# Ensure this model is available to your Groq account.
model = init_chat_model("groq:openai/gpt-oss-120b")

print("\nModel initialized successfully!")
print("\n Model Profie",model.profile)


# ============================================================
# EXAMPLE 1: PYDANTIC STRUCTURED OUTPUT
# USE CASE: E-COMMERCE PRODUCT EXTRACTION
# ============================================================

from pydantic import BaseModel, Field


class Product(BaseModel):
    """
    Pydantic automatically validates the output
    against the declared field types and constraints.
    """

    name: str = Field(
        description="Product name"
    )

    storage: str = Field(
        description="Product storage capacity"
    )

    price: float = Field(
        description="Product price in INR",
        gt=0
    )

    color: str = Field(
        description="Product color"
    )


def example_1_pydantic():

    print("\n========== EXAMPLE 1: PYDANTIC ==========")

    # Convert the chat model into a structured-output model.
    # The LLM is instructed to return data matching Product.
    structured_model = model.with_structured_output(
        Product,
    )

    # Provide unstructured text.
    response = structured_model.invoke(
        """
        Provide product details for Apple iPhone 12
        using your existing knowledge.

        Include a commonly available storage capacity,
        an estimated price in INR, and a color.
   
        """
    )

    # Response is a validated Product instance.
    print("\nComplete response:")
    print(response)

    # Access fields using dot notation.
    print("\nProduct name:", response.name)
    print("Storage:", response.storage)
    print("Price:", response.price)
    print("Color:", response.color)

    # Convert Pydantic object into a dictionary.
    print("\nDictionary:")
    print(response.model_dump())

    # Convert Pydantic object into JSON.
    print("\nJSON:")
    print(response.model_dump_json(indent=2))
    movie_example()

class Actor(BaseModel):
    name: str
    role: str
class Movie(BaseModel):
    """Expected structure of movie information."""

    title: str = Field(
        description="The title of the movie."
    )

    year: int = Field(
        description="The year the movie was released."
    )

    director: str = Field(
        description="The director of the movie."
    )
    cast: list[Actor]
    genres: list[str]
    rating: float = Field(
        description="The movie's rating out of 10.",
        ge=0,
        le=10
    )


def movie_example():

    print("\n========== MOVIE EXAMPLE ==========")

    # Tell the model to return output matching Movie.
    structured_model = model.with_structured_output(
        Movie,
        method="function_calling"
    )

    # The model can use its existing knowledge about Inception.
    response = structured_model.invoke(
        """
        Provide details about the movie Inception.

        Include its title, release year, director
        and rating out of 10.
        """
    )

    # Print complete Pydantic object.
    print("\nComplete response:")
    print(response)

    # Access individual fields.
    print("\nTitle:", response.title)
    print("Year:", response.year)
    print("Director:", response.director)
    print("Rating:", response.rating)

    # Convert to dictionary.
    print("\nDictionary:")
    print(response.model_dump())

    # Convert to JSON.
    print("\nJSON:")
    print(response.model_dump_json(indent=2))

# ============================================================
# EXAMPLE 2: NESTED PYDANTIC STRUCTURES
# USE CASE: RESUME / CANDIDATE EXTRACTION
# ============================================================


class Skill(BaseModel):
    """Represents one skill of a candidate."""

    name: str

    experience_years: float


class Education(BaseModel):
    """Represents candidate education details."""

    degree: str

    university: str

    graduation_year: int


class Candidate(BaseModel):
    """
    Main candidate model.

    This contains:
    - Normal fields
    - A list of nested Skill objects
    - A nested Education object
    """

    name: str

    skills: list[Skill]

    education: Education


def example_2_nested_pydantic():

    print("\n========== EXAMPLE 2: NESTED PYDANTIC ==========")

    structured_model = model.with_structured_output(
        Candidate,
        method="function_calling"
    )

    response = structured_model.invoke(
        """
        Extract candidate information:

        Rahul Sharma has 4 years of React experience
        and 2 years of Node.js experience.

        He completed B.Tech in Computer Science
        from XYZ University in 2022.
        """
    )

    print("\nComplete response:")
    print(response)

    # Access top-level field.
    print("\nCandidate name:", response.name)

    # Access nested Education object.
    print("Degree:", response.education.degree)
    print("University:", response.education.university)

    # Iterate through the list of nested Skill objects.
    print("\nSkills:")

    for skill in response.skills:
        print("Skill:", skill.name)
        print("Experience:", skill.experience_years)
        print("------------------------")

    # Convert complete nested structure to dictionary.
    print("\nDictionary:")
    print(response.model_dump())


# ============================================================
# EXAMPLE 3: TYPEDDICT STRUCTURED OUTPUT
# USE CASE: CUSTOMER SUPPORT CLASSIFICATION
# ============================================================

from typing_extensions import TypedDict, Annotated


class SupportTicket(TypedDict):
    """
    TypedDict defines the expected dictionary structure.

    Unlike Pydantic, TypedDict does not independently
    validate runtime values.
    """

    category: Annotated[
        str,
        "Category of the customer complaint"
    ]

    priority: Annotated[
        str,
        "Priority: low, medium or high"
    ]

    summary: Annotated[
        str,
        "Short summary of the complaint"
    ]

    requires_refund: Annotated[
        bool,
        "Whether the customer is requesting a refund"
    ]


def example_3_typeddict():

    print("\n========== EXAMPLE 3: TYPEDDICT ==========")

    structured_model = model.with_structured_output(
        SupportTicket,
        method="function_calling"
    )

    response = structured_model.invoke(
        """
        My payment was deducted,
        but my order was not placed.

        Please help me get my money back.
        """
    )

    print("\nComplete response:")
    print(response)

    # TypedDict output is a normal Python dictionary.
    # Therefore, access fields using dictionary notation.

    print("\nCategory:", response["category"])
    print("Priority:", response["priority"])
    print("Summary:", response["summary"])
    print("Requires refund:", response["requires_refund"])


# ============================================================
# EXAMPLE 4: DATACLASS STRUCTURED OUTPUT
# USE CASE: EMPLOYEE INFORMATION EXTRACTION
# ============================================================

from dataclasses import dataclass, asdict


@dataclass
class Employee:
    """
    Dataclass provides a lightweight Python data container.

    Python automatically generates methods such as
    __init__() and __repr__().
    """

    employee_id: int

    name: str

    department: str

    salary: float


def example_4_dataclass():

    print("\n========== EXAMPLE 4: DATACLASS ==========")

    structured_model = model.with_structured_output(
        Employee,
        method="function_calling"
    )

    structured_data = structured_model.invoke(
        """
        Extract employee information:

        Employee ID: 101
        Name: Amit
        Department: Engineering
        Salary: 85000
        """
    )

    # LangChain may return dataclass-schema output as a dictionary. Build the
    # dataclass explicitly so the following example can use attribute access.
    response = Employee(**structured_data)

    print("\nComplete response:")
    print(response)

    # Access dataclass fields using dot notation.
    print("\nEmployee ID:", response.employee_id)
    print("Name:", response.name)
    print("Department:", response.department)
    print("Salary:", response.salary)

    # asdict() converts a dataclass instance to a dictionary.
    print("\nDictionary:")
    print(asdict(response))


# ============================================================
# EXAMPLE 5: STRUCTURED OUTPUT WITH include_raw=True
# USE CASE: EXPENSE EXTRACTION AND DEBUGGING
# ============================================================


class Expense(BaseModel):
    """Represents an individual expense."""

    amount: float = Field(
        description="Expense amount in INR",
        gt=0
    )

    category: str = Field(
        description="Expense category"
    )

    description: str = Field(
        description="Expense description"
    )


def example_5_include_raw():

    print("\n========== EXAMPLE 5: INCLUDE RAW ==========")

    structured_model = model.with_structured_output(
        Expense,
        method="function_calling",
        include_raw=True
    )

    response = structured_model.invoke(
        """
        I spent Rs. 1200 on groceries yesterday.
        """
    )

    # include_raw=True returns a dictionary containing
    # raw response, parsed response and parsing error.

    raw_response = response["raw"]

    parsed_response = response["parsed"]

    parsing_error = response["parsing_error"]

    print("\nRAW RESPONSE:")
    print(raw_response)

    print("\nPARSED RESPONSE:")
    print(parsed_response)

    print("\nPARSING ERROR:")
    print(parsing_error)

    # Only access structured fields when parsing succeeds.
    if parsed_response is not None:

        print("\nExpense amount:", parsed_response.amount)

        print("Expense category:", parsed_response.category)

        print("Expense description:", parsed_response.description)

    else:
        print("Structured output parsing failed.")


# ============================================================
# EXAMPLE 6: CREATE_AGENT + TOOLSTRATEGY
# USE CASE: TRAVEL PLANNING
# ============================================================

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class TravelPlan(BaseModel):
    """
    Defines the structured response expected from the agent.
    """

    destination: str = Field(
        description="Recommended travel destination"
    )

    duration_days: int = Field(
        description="Number of travel days",
        gt=0
    )

    estimated_budget: float = Field(
        description="Estimated budget in INR",
        gt=0
    )

    activities: list[str] = Field(
        description="List of recommended activities"
    )


def example_6_agent_tool_strategy():

    print("\n========== EXAMPLE 6: AGENT + TOOLSTRATEGY ==========")

    # Create a LangChain agent.
    #
    # tools=[] means this agent has no external application
    # tools. ToolStrategy uses the model's tool-calling
    # capability to generate the structured response.

    agent = create_agent(
        model=model,
        tools=[],
        response_format=ToolStrategy(TravelPlan)
    )

    # Invoke the agent.
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": """
                Plan a 3-day budget trip to ooty for honeymoon
                with an estimated budget of Rs. 12000.

                Include sightseeing and food activities.
                """
            }
        ]
    })

    # Extract structured response from agent state.
    response = result["structured_response"]

    print("\nComplete structured response:")
    print(response)

    print("\nDestination:", response.destination)

    print("Duration:", response.duration_days, "days")

    print("Budget:", response.estimated_budget)

    print("\nActivities:")

    for activity in response.activities:
        print("-", activity)


# ============================================================
# 7. MAIN FUNCTION
# ============================================================

def main():

    print("\n============================================")
    print("LANGCHAIN V1 - STRUCTURED OUTPUT PRACTICE")
    print("============================================")

    # Uncomment the examples you want to execute.

    # example_1_pydantic()

    # example_2_nested_pydantic()

    # example_3_typeddict()

    # example_4_dataclass()

    # example_5_include_raw()

    # example_6_agent_tool_strategy()

    print("\n============================================")
    print("EXECUTION COMPLETED")
    print("============================================")


# Python entry point.
# This ensures main() runs only when this file
# is executed directly, not when imported elsewhere.

if __name__ == "__main__":
    main()


# ============================================================
# END OF STRUCTURED OUTPUT PRACTICE
# ============================================================
