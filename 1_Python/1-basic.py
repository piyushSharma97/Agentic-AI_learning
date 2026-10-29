#!/usr/bin/env python
# coding: utf-8

"""
Python Fundamentals: Syntax Rules, Case Sensitivity, Indentation & Line Structure

Why Python for AI and Data Science?
Python is the undisputed lingua franca of Artificial Intelligence and Machine Learning.
Its clean, readable syntax allows engineers to focus on mathematical models, data pipelines,
and prompt engineering rather than low-level memory management.

Core Syntax Rules Covered:
1. Case Sensitivity: Python treats lowercase and uppercase identifiers as completely different entities.
2. Indentation: Whitespace at the start of a line defines scope and code blocks (no curly braces {} needed!).
3. Line Continuation: Splitting long mathematical or string expressions across multiple physical lines.
4. Multiple Statements: Using semicolons (;) to execute multiple instructions in a single line.
"""

# --------------------------------------------------
# 1. Case Sensitivity in Python
# --------------------------------------------------
# Python is strictly case-sensitive.
# 'name', 'Name', and 'NAME' are three completely independent variables in memory.
# In the example below, assigning 'Sharma' to 'name' replaces the previous value 'Piyush'.

name = 'Piyush'
name = 'Sharma'  # Re-binding: 'name' now references 'Sharma'

print("Updated name:", name)

# Example of case sensitivity distinction:
Model = "GPT-4"
model = "Claude-3"
print(f"Model is '{Model}', but model is '{model}' (Case matters!)")

# --------------------------------------------------
# 2. Indentation as Block Structure
# --------------------------------------------------
# While languages like C++, Java, or JavaScript use curly braces { } to define code blocks,
# Python uses indentation (typically 4 spaces per nesting level).
# Consistent indentation is enforced by the Python interpreter; inconsistent spaces raise an `IndentationError`.

age = 32

# The indented block below 'if' only executes if the boolean condition (age > 30) evaluates to True.
if age > 30:
    print("Inside block: age is greater than 30 ->", age)

# Unindented line: Executes unconditionally after the if statement finishes
print("Outside block: current age is ->", age)

# --------------------------------------------------
# 3. Line Continuation (Backslash \)
# --------------------------------------------------
# In Python, the end of a physical line normally marks the end of a statement.
# When an expression is very long (e.g. chaining mathematical calculations or long prompt strings),
# a trailing backslash (\) explicitly tells Python: "Continue reading this statement on the next line."

total = 1 + 2 + 3 + 4 + 5 + \
    6

print("Total calculated using line continuation:", total)

# --------------------------------------------------
# 4. Multiple Statements on a Single Line (Semicolon ;)
# --------------------------------------------------
# Semicolons allow grouping multiple statements on a single physical line.
# Note: PEP 8 (Python style guide) strongly discourages semicolons in production code
# because separate lines improve readability and git diff clarity.
x = 5; y = 6; z = x * y
print("Result of x * y on single line:", z)

# --------------------------------------------------
# 5. Nested Indentation Blocks
# --------------------------------------------------
# Each additional level of indentation creates a sub-block.
# The inner block only runs if BOTH the outer condition and inner condition are met.

if True:
    print("Outer block: this is true and will print")
    if False:
        # This sub-block is skipped because its condition is False
        print("Inner block: this will NOT print")
    print("Outer block continues: this will also print")

print("Global scope: completely outside all indented blocks")

# ============================================================
# REVISION NOTES — 1-basic.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This introductory script introduces core Python syntax rules:
# variable reassignment, case sensitivity, block indentation, explicit line continuation,
# and nested conditional execution.
#
# KEY CONCEPTS USED:
# • Case Sensitivity: Identifier capitalization matters ('a' != 'A').
# • Indentation: 4 spaces define scope and execution hierarchy.
# • Re-binding: Assigning a new value to an existing variable name updates its pointer.
# • Line Continuation (\): Allows long mathematical or prompt statements to span lines cleanly.
#
# HOW IT CONNECTS TO THE PROJECT:
# Clean indentation and correct syntax are essential prerequisites for writing LangChain runnables,
# defining custom functions, and orchestrating AI workflows.
#
# THINGS TO REMEMBER:
# • Always use 4 spaces for indentation; never mix tabs and spaces in modern Python (PEP 8).
# • An unindented line signals the end of the preceding code block.
# • Prefer one clear statement per line over semicolon-separated statements.
#
# COMMON MISTAKES TO AVOID:
# • IndentationError: Inconsistent indentation levels within the same block.
# • Unintended re-binding: Accidentally overwriting a variable by reusing its name with different capitalization.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Observe how the nested `if True` and `if False` blocks control which `print()` statements execute.
# ============================================================
