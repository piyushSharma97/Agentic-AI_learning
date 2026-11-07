"""
=============================================================================
LANGGRAPH COMPONENT 3: STATE SCHEMA WITH PYDANTIC (RUNTIME VALIDATION)
=============================================================================
USE
Validate graph input before any node runs: types, ranges, and your own rules.
This is the schema to use when the caller is a web form, webhook, or model JSON.

WHY THIS APPROACH
TypedDict and dataclass only help the editor. They accept `age: "nope"`.
Pydantic is used because it checks the object as it is created, coerces
safe values (`"28"` becomes `28`), and raises ValidationError so a bad
request never enters `assess_fitness`. The cost is extra work on every input.
Skip it for a trusted script. Use it at the boundary of an API.

WHY PYDANTIC FOR STATE SCHEMAS?
In production AI applications, graph inputs often come from untrusted sources:
user web forms, external webhooks, LLM-generated JSON, or third-party APIs.

Neither `TypedDict` nor standard `@dataclass` validate types at runtime.
Pydantic `BaseModel` solves this by providing:
1. Strict Runtime Validation: Rejects malformed data immediately.
2. Type Coercion: Automatically converts compatible types (e.g. "42" -> 42).
3. Field Constraints: Validates ranges (e.g. `ge=0`), string lengths, regex patterns.
4. Custom Validators: Run Python logic to sanitize inputs before any node executes.

CRITICAL ARCHITECTURE NOTE:
Never name your local Python script `pydantic.py`! In Python, local filenames
take precedence over installed packages. Naming a file `pydantic.py` will shadow
the third-party Pydantic library and break all Pydantic imports across your project!
=============================================================================
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, ValidationError
from langgraph.graph import StateGraph, START, END

# Import shared helper
from config import print_mermaid, safe_run


# ---------------------------------------------------------------------------
# 1. DEFINE PYDANTIC STATE SCHEMA WITH CONSTRAINTS
# ---------------------------------------------------------------------------
class PlayerPydanticState(BaseModel):
    """
    Pydantic State Schema with strict runtime validation.
    """
    player_id: str = Field(..., description="Unique alphanumeric player identifier")
    name: str = Field(..., min_length=2, max_length=50, description="Player full name")
    age: int = Field(..., ge=5, le=100, description="Age must be between 5 and 100")
    game: Literal["cricket", "badminton"] = Field(..., description="Chosen sport")
    stamina: int = Field(default=100, ge=0, le=100, description="Stamina score (0-100)")
    status: str = Field(default="Registered", description="Current status message")

    # Use: a check that Field() cannot express, here the "PL-" prefix.
    # Why: Field(ge=5) covers ranges. A prefix rule is your own logic, so it
    # lives in a validator that runs before the graph starts.
    @field_validator("player_id")
    @classmethod
    def validate_player_id(cls, value: str) -> str:
        upper_val = value.upper()
        if not upper_val.startswith("PL-"):
            raise ValueError(f"player_id must start with prefix 'PL-'. Received: '{value}'")
        return upper_val


# ---------------------------------------------------------------------------
# 2. DEFINE NODE FUNCTIONS
# ---------------------------------------------------------------------------
def assess_fitness(state: PlayerPydanticState) -> dict:
    print(f"\n[Node: assess_fitness] Checking vitals for {state.name} (Age: {state.age})...")
    stamina_burn = 10 if state.age < 40 else 15
    new_stamina = max(0, state.stamina - stamina_burn)
    return {
        "stamina": new_stamina,
        "status": f"{state.name} cleared fitness check with {new_stamina}% stamina."
    }


def execute_match(state: PlayerPydanticState) -> dict:
    print(f"[Node: execute_match] Playing {state.game.upper()} match for {state.name}...")
    burn = 30 if state.game == "cricket" else 20
    final_stamina = max(0, state.stamina - burn)
    return {
        "stamina": final_stamina,
        "status": f"Match completed! Remaining stamina: {final_stamina}%."
    }


# ---------------------------------------------------------------------------
# 3. BUILD THE GRAPH
# ---------------------------------------------------------------------------
def build_pydantic_graph():
    builder = StateGraph(PlayerPydanticState)

    builder.add_node("assess_fitness", assess_fitness)
    builder.add_node("execute_match", execute_match)

    builder.add_edge(START, "assess_fitness")
    builder.add_edge("assess_fitness", "execute_match")
    builder.add_edge("execute_match", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# 4. EXECUTION & VALIDATION TESTS
# ---------------------------------------------------------------------------
def main():
    def demo():
        graph = build_pydantic_graph()
        print_mermaid(graph, "Pydantic State Validation Graph")

        print("--- Test 1: Valid Input (Everything Passes Validation) ---")
        valid_input = {
            "player_id": "pl-101",
            "name": "Virat",
            "age": 35,
            "game": "cricket",
            "stamina": 95
        }
        res = graph.invoke(valid_input)
        # res is returned as a dict representing the final state
        p_id = res.get("player_id", getattr(res, "player_id", "N/A"))
        name = res.get("name", getattr(res, "name", "N/A"))
        game = res.get("game", getattr(res, "game", "N/A"))
        stamina = res.get("stamina", getattr(res, "stamina", "N/A"))
        status = res.get("status", getattr(res, "status", "N/A"))
        print(f"Success! Player ID: {p_id} (Auto-uppercased by validator)")
        print(f"Name: {name} | Game: {game} | Stamina: {stamina}%")
        print(f"Final Status: {status}\n")

        print("--- Test 2: Automatic Type Coercion (String Age -> Integer) ---")
        # Notice age is a string "28". Pydantic will coerce it to int 28
        coerced_input = {
            "player_id": "pl-102",
            "name": "Saina",
            "age": "28",  # String converted to int!
            "game": "badminton"
        }
        res_coerced = graph.invoke(coerced_input)
        c_age = res_coerced.get("age", getattr(res_coerced, "age", "N/A"))
        print(f"Coerced Age: {c_age} (Type: {type(c_age).__name__})\n")

        print("--- Test 3: Runtime Rejection (Invalid ID Prefix) ---")
        try:
            graph.invoke({
                "player_id": "INVALID-ID",  # Fails validator: missing 'PL-' prefix
                "name": "John",
                "age": 25,
                "game": "cricket"
            })
        except ValidationError as e:
            print("Pydantic caught invalid player_id immediately!")
            print(f"Error Message:\n  -> {e.errors()[0]['msg']}\n")

        print("--- Test 4: Runtime Rejection (Out of Range Age & Invalid Sport) ---")
        try:
            graph.invoke({
                "player_id": "PL-999",
                "name": "Kid",
                "age": 2,  # Violates ge=5 constraint!
                "game": "football"  # Violates Literal['cricket', 'badminton']!
            })
        except ValidationError as e:
            print(f"Pydantic blocked execution with {len(e.errors())} validation errors:")
            for err in e.errors():
                print(f"  - Field '{err['loc'][0]}': {err['msg']}")

    safe_run("LangGraph State Schema with Pydantic Runtime Validation", demo)


if __name__ == "__main__":
    main()
