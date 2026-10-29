#!/usr/bin/env python
# coding: utf-8

"""
Python Variables, User Input, and Type Casting

What is a Variable in Python?
A variable is a symbolic name that acts as a reference or pointer to an object stored in computer memory (RAM).
Python is dynamically typed: you do NOT need to declare a variable's type (like 'int x' in C or Java).
The Python interpreter infers the data type automatically at runtime based on the assigned value.

Topics Covered:
1. Variable declaration and assignment.
2. Python naming conventions (PEP 8 standards).
3. Reading user input from the console using input().
4. Implicit type conversion (automatic promotion).
5. Explicit type conversion (type casting functions like int(), float(), str()).
"""

# --------------------------------------------------
# 1. Declaring and Assigning Variables
# --------------------------------------------------
# Variables are created the moment you first assign a value to them using the assignment operator (=).

age = 30            # Integer (int): whole number
height = 5.4        # Floating point (float): decimal number
name = "Piyush"     # String (str): sequence of text characters
is_employee = True  # Boolean (bool): truth value (True or False)

print("Employee Status:", is_employee)
print("Employee Name:", name)
print("Age:", age)
print("Height:", height)

# --------------------------------------------------
# 2. Variable Naming Rules & Best Practices (PEP 8)
# --------------------------------------------------
# • Must start with a letter (a-z, A-Z) or an underscore (_).
# • Cannot start with a number (e.g. '1name' is INVALID).
# • Can only contain alphanumeric characters and underscores (a-z, 0-9, _).
# • Cannot use Python reserved keywords (e.g. 'if', 'for', 'class', 'import').
# • Python variables are case-sensitive ('my_var' != 'My_Var').
# • Best practice: Use 'snake_case' for variable names (e.g., user_query, chunk_size, api_key).

user_model_name = "gpt-4o"
_internal_counter = 42

# --------------------------------------------------
# 3. Capturing User Input with input()
# --------------------------------------------------
# The input() function pauses execution and waits for the user to type something in the console.
# CRITICAL TAKEAWAY FOR AI DEVELOPERS:
# input() ALWAYS returns a string (str), even if the user types numeric digits (e.g., '25')!

num = input('Enter a number: ')

print('You Entered:', num)
print('Data type of num:', type(num))  # Always <class 'str'>!

# --------------------------------------------------
# 4. Implicit Type Conversion (Automatic Coercion)
# --------------------------------------------------
# In arithmetic operations involving mixed types, Python automatically promotes the narrower type
# to the wider type to prevent precision loss.
# Adding an integer (123) to a float (1.23) implicitly converts the result to a float (124.23).

integer_number = 123
float_number = 1.23

new_number = integer_number + float_number

print("\nImplicit Conversion Result:", new_number)
print("Data Type of resulting value:", type(new_number))  # <class 'float'>

# --------------------------------------------------
# 5. Explicit Type Conversion (Type Casting)
# --------------------------------------------------
# Because input() returns strings, you cannot perform mathematical addition directly with numbers:
# "12" + 23 would trigger a TypeError!
# You must explicitly convert (cast) the string to an integer using int().

num_string = '12'
num_integer = 23

print("\nData type of num_string before Type Casting:", type(num_string))

# Explicitly cast string to integer
num_string_converted = int(num_string)

print("Data type of num_string after Type Casting:", type(num_string_converted))

# Now arithmetic addition works correctly: 23 + 12 = 35
num_sum = num_integer + num_string_converted

print("Calculated Sum:", num_sum)
print("Data type of num_sum:", type(num_sum))

# ============================================================
# REVISION NOTES — 1.1-variable.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script covers variable creation, dynamic type inference, identifier naming conventions,
# reading string inputs via input(), and converting between strings, integers, and floats.
#
# KEY CONCEPTS USED:
# • Dynamic Typing: Python automatically associates data types with values at runtime.
# • input(): Always captures console input as a string (`str`).
# • Implicit Casting: Python automatically promotes an `int` to a `float` in mixed operations.
# • Explicit Casting (`int()`, `float()`, `str()`): Manually transforming values from one type to another.
#
# HOW IT CONNECTS TO THE PROJECT:
# Variables hold critical settings in every AI app: `chunk_size = 500` (int), `temperature = 0.7` (float),
# `query = "What is RAG?"` (str), and `streaming = True` (bool).
#
# THINGS TO REMEMBER:
# • Raw input from users, web scrapers, or API JSON calls is almost always text (strings);
#   always cast numeric parameters (e.g. `k` nearest neighbors) to `int`.
# • Use meaningful snake_case names for variables (`api_key` rather than `a`).
#
# COMMON MISTAKES TO AVOID:
# • TypeError: Adding a string and an integer directly (e.g. `"Token count: " + 50`).
# • ValueError: Attempting to cast non-numeric text to an integer (e.g. `int("hello")`).
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at `num = input()` -> inspect its type with `type(num)` -> cast with `int(num)`.
# ============================================================
