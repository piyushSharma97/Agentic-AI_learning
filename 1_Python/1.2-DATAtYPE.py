#!/usr/bin/env python
# coding: utf-8

"""
Python Data Types: Integers, Floats, Booleans, Strings & Type Checking

Why Data Types Matter:
Data types tell the Python interpreter:
1. What kind of value a variable holds (whole number, decimal, text, truth value).
2. What operations can be legally performed on that value (e.g., you can divide numbers, but not strings).
3. How much memory should be allocated for storage.

Python is Strongly Typed:
Python will NOT automatically convert a number into text when concatenating with strings (unlike JavaScript or PHP).
This prevents subtle bugs in calculations and data pipelines!
"""

# --------------------------------------------------
# 1. Integer (int)
# --------------------------------------------------
# Whole numbers without fractional or decimal components (can be positive, negative, or zero).
# In Python 3, integers have arbitrary precision (they can grow as large as available RAM).

age = 30
print("Value of age:", age)
print("Type of age:", type(age))  # <class 'int'>

# --------------------------------------------------
# 2. Floating-Point Number (float)
# --------------------------------------------------
# Numbers that contain a decimal point or are written in scientific notation (e.g., 2.5e-3).
# In AI, floats are ubiquitous: model weights, embedding vectors, probabilities, and temperatures are all floats!

length = 5.6
print("\nValue of length:", length)
print("Type of length:", type(length))  # <class 'float'>

# --------------------------------------------------
# 3. Boolean (bool)
# --------------------------------------------------
# Represents one of two truth states: True or False (capitalization is required!).
# Internally in Python, `bool` is a subclass of `int` where True == 1 and False == 0.

is_bird = True
print("\nValue of is_bird:", is_bird)
print("Type of is_bird:", type(is_bird))  # <class 'bool'>

# --------------------------------------------------
# 4. Evaluating Expressions into Booleans
# --------------------------------------------------
# Comparison operations (==, !=, <, >) automatically evaluate to a boolean value.

a = 4
b = 4

# Comparison: Does 'a' equal 'b'? Evaluates to True.
comparison_result = (a == b)
print(f"\nDoes {a} == {b}? -> {comparison_result}")
print("Type of comparison_result:", type(comparison_result))

# --------------------------------------------------
# 5. Strong Typing: Handling Common TypeErrors
# --------------------------------------------------
# Common beginner pitfall:
# Attempting to concatenate text with a raw number:
#   res = "hello " + 5
# Raises: `TypeError: can only concatenate str (not "int") to str`
#
# Because Python is strongly typed, you must explicitly convert the integer to a string using str():

res_correct = "hello " + str(5)

print("\nResult of explicit str conversion concatenation:", res_correct)
print("Type of res_correct:", type(res_correct))

# Alternative modern best practice (Python 3.6+ f-strings):
number_of_tokens = 5
f_string_example = f"Processed {number_of_tokens} tokens"
print("Clean concatenation using f-strings:", f_string_example)

# ============================================================
# REVISION NOTES — 1.2-DATAtYPE.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script introduces Python's fundamental scalar data types: integers (`int`), floating-point numbers (`float`),
# booleans (`bool`), and strings (`str`). It highlights Python's strong typing system and shows how to convert
# numbers to text before string concatenation.
#
# KEY CONCEPTS USED:
# • `type()`: Built-in function to inspect the runtime class/type of any object.
# • `int`: Whole numbers (used for chunk sizes, k-nearest neighbors, token counts).
# • `float`: Decimal numbers (used for embeddings, temperature, distance scores).
# • `bool`: Logical values (`True` or `False`), essential for conditionals and feature flags.
# • Strong Typing: Python enforces strict type compatibility and refuses to implicitly add numbers to strings.
#
# HOW IT CONNECTS TO THE PROJECT:
# In every LangChain component, data types dictate behavior:
# - `ChatOpenAI(temperature=0.2)` -> float
# - `retriever.as_retriever(search_kwargs={"k": 3})` -> int
# - `query = "What is AI?"` -> str
# - `streaming = True` -> bool
#
# THINGS TO REMEMBER:
# • Always write `True` and `False` with capital first letters; `true` and `false` are not keywords in Python.
# • Use modern f-strings (`f"Count: {x}"`) instead of manual `+ str(x)` concatenation for cleaner code.
#
# COMMON MISTAKES TO AVOID:
# • Writing `"5" + 5` and expecting `10` or `"10"`; Python raises a `TypeError`.
# • Confusing the assignment operator `=` with the equality comparison operator `==`.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Compare the error-prone `"hello" + 5` with the working `f"hello {5}"` or `"hello" + str(5)`.
# ============================================================
