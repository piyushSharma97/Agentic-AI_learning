"""
=============================================================================
LANGGRAPH COMPONENT 3: STATE SCHEMA WITH PYDANTIC (RUNTIME VALIDATION)
=============================================================================
TOPIC OVERVIEW:
While Topics 01 (TypedDict) and 02 (Dataclass) are great for internal scripts,
neither enforces data types at runtime.

In production AI applications, graph inputs often originate from untrusted sources:
user web forms, external webhooks, LLM-generated JSON, or third-party APIs.
Pydantic `BaseModel` solves this by providing:
1. Strict Runtime Validation: Rejects malformed data before any node executes.
2. Type Coercion: Automatically converts compatible types (e.g. "28" -> 28).
3. Field Constraints: Validates bounds (`ge=5, le=100`), lengths, and regex.
4. Custom Validators: Runs Python domain rules via `@field_validator`.

FLOW OF EXECUTION:
    [START] 
       │
       ▼ (Validates player_id, age, game, stamina)
 [assess_fitness] (Calculates stamina burn based on age)
       │
       ▼
 [execute_match]  (Calculates stamina burn based on game)
       │
       ▼
     [END]

CRITICAL SHADOWING WARNING:
Never name your local Python file `pydantic.py`! Python resolves local files
before installed packages. Naming a file `pydantic.py` will shadow the official
library and break all Pydantic imports across your environment!
=============================================================================
"""

import sys
from pathlib import Path
from typing import Literal

# Add parent directory (10_LangGrapph_Component) to sys.path so config can be imported directly
_parent_dir = str(Path(__file__).resolve().parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from pydantic import BaseModel, Field, field_validator, ValidationError
from langgraph.graph import StateGraph, START, END
from config import print_mermaid, safe_run


# ---------------------------------------------------------------------------
# STEP 1: DEFINE PYDANTIC STATE SCHEMA WITH RUNTIME CONSTRAINTS
# ---------------------------------------------------------------------------
class PlayerPydanticState(BaseModel):
    """
    Pydantic State Schema with strict runtime validation rules:
    - player_id: Must start with prefix 'PL-' (enforced by custom validator)
    - name: String between 2 and 50 characters
    - age: Integer between 5 and 100
    - game: Strictly 'cricket' or 'badminton'
    - stamina: Integer between 0 and 100 (defaults to 100)
    - status: Operational message string
    """
    player_id: str = Field(..., description="Unique alphanumeric player identifier")
    name: str = Field(..., min_length=2, max_length=50, description="Player full name")
    age: int = Field(..., ge=5, le=100, description="Age must be between 5 and 100")
    game: Literal["cricket", "badminton"] = Field(..., description="Chosen sport")
    stamina: int = Field(default=100, ge=0, le=100, description="Stamina score (0-100)")
    status: str = Field(default="Registered", description="Current status message")

    @field_validator("player_id")
    @classmethod
    def validate_player_id(cls, value: str) -> str:
        """
        Custom validator ensuring every player ID begins with 'PL-'.
        Automatically converts lowercase prefixes like 'pl-101' to uppercase 'PL-101'.
        """
        upper_val = value.upper()
        if not upper_val.startswith("PL-"):
            raise ValueError(f"player_id must start with prefix 'PL-'. Received: '{value}'")
        return upper_val


# ---------------------------------------------------------------------------
# STEP 2: DEFINE NODE FUNCTIONS
# ---------------------------------------------------------------------------
def assess_fitness(state: PlayerPydanticState) -> dict:
    """
    Node 1: Evaluates fitness based on age.
    Older players burn more stamina during fitness assessment.
    """
    print(f"\n[Node: assess_fitness] Checking vitals for {state.name} (Age: {state.age})...")
    stamina_burn = 10 if state.age < 40 else 15
    new_stamina = max(0, state.stamina - stamina_burn)
    return {
        "stamina": new_stamina,
        "status": f"{state.name} cleared fitness check with {new_stamina}% stamina."
    }


def execute_match(state: PlayerPydanticState) -> dict:
    """
    Node 2: Simulates match execution and calculates remaining stamina.
    Cricket consumes 30% stamina; Badminton consumes 20% stamina.
    """
    print(f"[Node: execute_match] Playing {state.game.upper()} match for {state.name}...")
    burn = 30 if state.game == "cricket" else 20
    final_stamina = max(0, state.stamina - burn)
    return {
        "stamina": final_stamina,
        "status": f"Match completed! Remaining stamina: {final_stamina}%."
    }


# ---------------------------------------------------------------------------
# STEP 3: ASSEMBLE AND COMPILE THE STATEGRAPH
# ---------------------------------------------------------------------------
def build_pydantic_graph():
    """
    Wires up the sequential pipeline: START -> assess_fitness -> execute_match -> END
    """
    builder = StateGraph(PlayerPydanticState)

    builder.add_node("assess_fitness", assess_fitness)
    builder.add_node("execute_match", execute_match)

    builder.add_edge(START, "assess_fitness")
    builder.add_edge("assess_fitness", "execute_match")
    builder.add_edge("execute_match", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# STEP 4: EXECUTION & VALIDATION DEMONSTRATIONS
# ---------------------------------------------------------------------------
def main():
    def demo():
        graph = build_pydantic_graph()
        print_mermaid(graph, "Pydantic State Validation Graph")

        # Test Case 1: Valid Input (passes all checks)
        print("--- Test 1: Valid Input (Everything Passes Validation) ---")
        valid_input = {
            "player_id": "pl-101",
            "name": "Virat",
            "age": 35,
            "game": "cricket",
            "stamina": 95
        }
        res = graph.invoke(valid_input)
        p_id = res.get("player_id", getattr(res, "player_id", "N/A"))
        name = res.get("name", getattr(res, "name", "N/A"))
        game = res.get("game", getattr(res, "game", "N/A"))
        stamina = res.get("stamina", getattr(res, "stamina", "N/A"))
        status = res.get("status", getattr(res, "status", "N/A"))
        print(f"Success! Player ID: {p_id} (Auto-uppercased by validator)")
        print(f"Name: {name} | Game: {game} | Stamina: {stamina}%")
        print(f"Final Status: {status}\n")

        # Test Case 2: Automatic Type Coercion (string "28" -> integer 28)
        print("--- Test 2: Automatic Type Coercion (String Age -> Integer) ---")
        coerced_input = {
            "player_id": "pl-102",
            "name": "Saina",
            "age": "28",  # String number automatically converted to int!
            "game": "badminton"
        }
        res_coerced = graph.invoke(coerced_input)
        c_age = res_coerced.get("age", getattr(res_coerced, "age", "N/A"))
        print(f"Coerced Age: {c_age} (Type: {type(c_age).__name__})\n")

        # Test Case 3: Runtime Rejection (Invalid ID Prefix)
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

        # Test Case 4: Runtime Rejection (Out of Range Age & Invalid Sport)
        print("--- Test 4: Runtime Rejection (Out of Range Age & Invalid Sport) ---")
        try:
            graph.invoke({
                "player_id": "PL-999",
                "name": "Kid",
                "age": 2,          # Violates ge=5 constraint!
                "game": "football" # Violates Literal['cricket', 'badminton']!
            })
        except ValidationError as e:
            print(f"Pydantic blocked execution with {len(e.errors())} validation errors:")
            for err in e.errors():
                print(f"  - Field '{err['loc'][0]}': {err['msg']}")

    safe_run("LangGraph State Schema with Pydantic Runtime Validation", demo)


if __name__ == "__main__":
    main()


# ============================================================
# REVISION NOTES — Topic 03: Pydantic State Schema
# ============================================================
#
# WHAT THIS FILE DOES:
# Demonstrates defining LangGraph state using Pydantic `BaseModel`.
# Shows how Pydantic enforces strict runtime validation, field range checks,
# custom field validators, and safe type coercion before any node executes.
#
# KEY CONCEPTS USED:
# • BaseModel: Pydantic base class for runtime schema definition.
# • Field(ge=..., le=...): Enforces numeric and string length boundaries.
# • @field_validator: Custom business logic for sanitizing and validating fields.
# • Type Coercion: Automatically casts compatible types like string `"28"` to integer `28`.
# • ValidationError: Exception raised if incoming data violates schema rules.
#
# HOW IT CONNECTS TO THE PROJECT:
# Pydantic is the gold standard for production LangGraph applications where
# graphs receive inputs from web apps, mobile clients, webhooks, or third-party APIs.
#
# THINGS TO REMEMBER:
# • NEVER name your script `pydantic.py` (it will shadow the official package).
# • Pydantic validation adds a tiny amount of CPU overhead compared to TypedDict,
#   which is well worth the security and crash prevention in production.
#
# COMMON MISTAKES TO AVOID:
# • Assuming TypedDict or Dataclass will catch bad types at runtime.
# • Forgetting default values on optional fields in Pydantic.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at `PlayerPydanticState` -> notice the `ge=5` and `@field_validator` ->
# check Test 3 and Test 4 to see how invalid inputs are blocked immediately.
# ============================================================
