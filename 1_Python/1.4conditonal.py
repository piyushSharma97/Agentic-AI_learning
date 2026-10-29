#!/usr/bin/env python
# coding: utf-8

"""
Python Flow Control: Conditional Branching (if, elif, else)

Why Conditionals are Central to AI Agents:
AI agents are decision engines. While LLMs generate natural language responses, the agentic runtime
around the LLM relies heavily on conditional statements (if, elif, else) to make deterministic choices:
- "If user asks to search the web -> call web search tool."
- "If token count > 1200 -> trigger message trimming."
- "If distance score < 0.5 -> include chunk in RAG context."
- "If error occurred -> retry or fallback to secondary model."

Syntax Pattern:
if condition_1:
    # Runs ONLY if condition_1 is True
elif condition_2:
    # Runs if condition_1 was False, BUT condition_2 is True
else:
    # Runs if NONE of the above conditions were True
"""

# --------------------------------------------------
# 1. Basic 'if' Statement (Single Branch)
# --------------------------------------------------
# The body of an 'if' statement executes if and only if its condition evaluates to True.
# If False, Python skips the indented block completely.

# Reminder: input() returns a string (str), so we must convert to int before numeric comparison!
user_input_str = input("Enter your age (e.g., 20): ")
age = int(user_input_str)

# Condition: Is age greater than or equal to 18?
if age >= 18:
    print("Condition met: You are an adult. Access granted.")

print("Statement after 'if' block always executes.\n")

# --------------------------------------------------
# 2. Dual-Branch Decision: 'if...else'
# --------------------------------------------------
# 'else' provides a fallback path when the 'if' condition evaluates to False.
# Exactly ONE of the two branches is guaranteed to execute.

voting_age = int(input("Enter age for voting verification: "))

if voting_age >= 18:
    print("Eligible to vote!")
else:
    print("Underage: Not eligible to vote yet.")

# --------------------------------------------------
# 3. Compound Conditions with Boolean 'and'
# --------------------------------------------------
# Both conditions must evaluate to True simultaneously for the block to execute.
# (This simulates credential verification or multi-factor authentication logic).

# Stored reference values
username_db = "admin"
password_db = "sparrow@123"

# Capture user credentials from console
username_input = input("\nEnter username: ")
password_input = input("Enter password: ")

# Check equality of BOTH username and password
if (username_input == username_db) and (password_input == password_db):
    print("Authentication successful: Welcome back, Administrator!")
else:
    print("Authentication failed: Invalid username or password.")

# --------------------------------------------------
# 4. Multi-Way Decision Ladder: 'if...elif...else'
# --------------------------------------------------
# When evaluating multiple mutually exclusive outcomes, use 'elif' (short for 'else if').
# Python evaluates conditions sequentially from top to bottom.
# The FIRST condition that evaluates to True executes its block; all subsequent elif/else blocks are SKIPPED!

test_age = int(input("\nEnter age for category classification: "))

if test_age < 0:
    print("Invalid Input: Age cannot be negative.")
elif test_age < 13:
    print("Category: Child")
elif test_age < 20:
    print("Category: Teenager")
elif test_age < 65:
    print("Category: Adult")
else:
    print("Category: Senior Citizen")

# --------------------------------------------------
# 5. Agentic AI Context: Intent Routing with Conditionals
# --------------------------------------------------
# In AI agent development, conditionals are used to route user commands:
user_command = "/new"

if user_command == "/new":
    print("\n[AI Agent]: Memory reset. New conversation session initialized.")
elif user_command.startswith("/language"):
    print("[AI Agent]: Switching target generation language.")
elif user_command in {"exit", "quit"}:
    print("[AI Agent]: Session terminated.")
else:
    print("[AI Agent]: Passing query to LLM reasoning loop.")

# ============================================================
# REVISION NOTES — 1.4conditonal.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script demonstrates decision-making in Python using conditional branching:
# single-branch `if`, dual-branch `if-else`, compound boolean criteria (`and`),
# and multi-tier categorization using `if-elif-else`.
#
# KEY CONCEPTS USED:
# • Conditional Branching: Directing execution flow based on runtime truth values.
# • Mutual Exclusivity: In an `if-elif-else` ladder, once any branch executes, the remaining branches are bypassed.
# • Type Conversion: Converting raw string input (`input()`) to integers (`int()`) before inequality comparisons.
# • Compound Logic: Using `and` to enforce multiple simultaneous constraints.
#
# HOW IT CONNECTS TO THE PROJECT:
# Conditionals appear across the codebase:
# - Validating API keys: `if not api_key: raise RuntimeError(...)`
# - Fallback file paths: `if not file_path.exists(): ...`
# - Slash command routing: `if text == "/new": session_id = str(uuid4())`
# - Message trimming: `if len(tokens) > max_tokens: trim(...)`
#
# THINGS TO REMEMBER:
# • The colon `:` at the end of `if`, `elif`, and `else` lines is mandatory syntax.
# • Indentation inside each branch must be consistent (4 spaces).
# • An `else` block has no condition; it catches everything that failed earlier checks.
#
# COMMON MISTAKES TO AVOID:
# • Forgetting `int()` on `input()`, causing `TypeError: '>=' not supported between instances of 'str' and 'int'`.
# • Using assignment `=` instead of equality comparison `==`.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at the `if-elif-else` ladder in Section 4: conditions are checked top-to-bottom until a match is found.
# ============================================================
