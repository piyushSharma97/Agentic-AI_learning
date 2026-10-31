#!/usr/bin/env python
# coding: utf-8

"""
Conditional statements: if, else, elif, and a nested if.

A condition is an expression that is true or false.
The indented block under if runs only when that expression is true.
elif tries another test when the earlier ones were false.
else runs when every test above it was false.
A nested if is a second decision inside a block that already ran.

JavaScript comparison:
- Python if / elif / else. JavaScript if / else if / else.
- Python uses a colon and indentation. JavaScript uses parentheses and curly braces.
- Python and / or. JavaScript && / ||.
- Python input() returns text, so int(input()) converts it before a numeric test.
  In a browser, Number(prompt()) is the rough match.
"""

# --------------------------------------------------
# 1. if by itself
# --------------------------------------------------
# There is no else here. If age is under 18, this block is skipped
# and the program simply continues.

age = 20

if age >= 18:
    print("You are allowed to vote in the elections")

# This line compares age and 18, then throws the True/False result away.
# A comparison does not print and does not branch unless it sits in an if,
# a while, or an assignment. JavaScript has the same rule.

age >= 18

# --------------------------------------------------
# 2. if / else — exactly one of the two blocks runs
# --------------------------------------------------

age = 16

if age >= 18:
    print("You are eligible for voting")
else:
    print("You are a minor")

# --------------------------------------------------
# 3. Nested if — a second question inside the first true branch
# --------------------------------------------------
# input() always returns a string. int() turns that string into a number
# so > and % are numeric tests, not text tests.

num = int(input("Enter the number"))

if num > 0:
    print("The number is positive")
    # This inner test runs only after we already know num is positive.
    # % 2 is the remainder. 0 means the number divides by 2 evenly.
    if num % 2 == 0:
        print("The number is even")
    else:
        print("The number is odd")
else:
    print("The number is zero or negative")
