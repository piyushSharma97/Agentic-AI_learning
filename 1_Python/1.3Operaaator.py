#!/usr/bin/env python
# coding: utf-8

"""
Python Operators Masterclass: Arithmetic, Comparison, Logical, Bitwise & Assignment

What is an Operator?
Operators are special symbols and keywords that perform computations on values and variables (called operands).
Python provides five primary families of operators:
1. Arithmetic Operators: Mathematical calculations (addition, floor division, modulo, powers).
2. Comparison (Relational) Operators: Evaluating relationships between values to return True or False.
3. Logical Operators: Combining boolean conditions (and, or, not).
4. Bitwise Operators: Manipulating raw binary bits directly (useful in cryptography, flags, and systems).
5. Assignment Operators: Updating variable values in place with shorthand arithmetic.
"""

# ==================================================
# 1. Arithmetic Operators
# ==================================================
# Arithmetic operators calculate numeric values.
# Key distinction to master:
# • `/`  (True Division): Always returns a `float`, even if operands divide evenly (e.g. 15 / 3 = 5.0).
# • `//` (Floor Division): Divides and drops the decimal part, rounding down towards negative infinity.
# • `%`  (Modulo): Returns the remainder of division (useful for checking even/odd or batching).
# • `**` (Exponentiation): Calculates power (x ** y means x raised to power y).

x = 15
y = 4

print("--- 1. Arithmetic Operators (x=15, y=4) ---")
print('x + y =', x + y)    # Addition: 19
print('x - y =', x - y)    # Subtraction: 11
print('x * y =', x * y)    # Multiplication: 60
print('x / y =', x / y)    # True Division (returns float): 3.75
print('x // y =', x // y)  # Floor Division (truncates decimals): 3
print('x % y =', x % y)    # Modulo (remainder of 15 / 4): 3
print('x ** y =', x ** y)  # Power (15^4): 50625

# ==================================================
# 2. Comparison (Relational) Operators
# ==================================================
# Compare two values and produce a boolean result (`True` or `False`).
# Critical in AI agent decision trees, input validation, and score thresholding.

x = 10
y = 12

print("\n--- 2. Comparison Operators (x=10, y=12) ---")
print('x > y  is', x > y)    # Greater than: False
print('x < y  is', x < y)    # Less than: True
print('x == y is', x == y)   # Equal to: False (Compare with ==, NOT single =)
print('x != y is', x != y)   # Not equal to: True
print('x >= y is', x >= y)   # Greater than or equal to: False
print('x <= y is', x <= y)   # Less than or equal to: True

# ==================================================
# 3. Logical (Boolean) Operators
# ==================================================
# Used to combine conditional expressions:
# • `and`: True ONLY if BOTH operands evaluate to True.
# • `or`:  True if AT LEAST ONE operand evaluates to True.
# • `not`: Reverses the boolean truth state (inverts True to False and vice versa).

x = True
y = False

print("\n--- 3. Logical Operators (x=True, y=False) ---")
print('x and y is', x and y)  # False (because y is False)
print('x or y  is', x or y)   # True (because x is True)
print('not x   is', not x)    # False (reverses True)

# Practical example in RAG:
has_api_key = True
tokens_available = True
is_rate_limited = False

can_query_llm = has_api_key and tokens_available and (not is_rate_limited)
print("Can system query LLM?", can_query_llm)  # True

# ==================================================
# 4. Bitwise Operators
# ==================================================
# Work directly on the individual binary bits (0s and 1s) of integer values.
# Symbols:
# • `&`  (AND): 1 if both bits are 1
# • `|`  (OR): 1 if either bit is 1
# • `^`  (XOR): 1 if bits are different
# • `~`  (NOT): Inverts all bits (two's complement: ~n = -(n + 1))
# • `>>` (Right Shift): Shifts bits right (equivalent to integer division by powers of 2)
# • `<<` (Left Shift): Shifts bits left (equivalent to multiplication by powers of 2)

print("\n--- 4. Bitwise Operators ---")
# 12 in binary: 0000 1100
# 5  in binary: 0000 0101
a_bin = 12
b_bin = 5

print('12 & 5 =', a_bin & b_bin)   # 4  (0000 0100)
print('12 | 5 =', a_bin | b_bin)   # 13 (0000 1101)
print('12 ^ 5 =', a_bin ^ b_bin)   # 9  (0000 1001)
print('~12    =', ~a_bin)          # -13
print('16 >> 2 =', 16 >> 2)        # 4  (16 / 2^2)
print('3 << 2  =', 3 << 2)         # 12 (3 * 2^2)

# ==================================================
# 5. Compound Assignment Operators
# ==================================================
# Shorthand notation that performs an operation and immediately reassigns the result
# to the left-hand variable (e.g. `x += 5` is identical to `x = x + 5`).

print("\n--- 5. Compound Assignment Operators ---")
val = 10
print(f"Initial: {val}")

val += 5   # val = val + 5 -> 15
print(f"After += 5: {val}")

val -= 3   # val = val - 3 -> 12
print(f"After -= 3: {val}")

val *= 2   # val = val * 2 -> 24
print(f"After *= 2: {val}")

val /= 4   # val = val / 4 -> 6.0 (Note: /= always converts to float!)
print(f"After /= 4: {val} (converted to float)")

val //= 2  # val = val // 2 -> 3.0
print(f"After //= 2: {val}")

val %= 2   # val = val % 2 -> 1.0 (remainder)
print(f"After %= 2: {val}")

val = 2
val **= 4  # val = 2^4 -> 16
print(f"After **= 4: {val}")

# ============================================================
# REVISION NOTES — 1.3Operaaator.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script is a reference guide to Python's five core operator families:
# arithmetic, comparison, logical, bitwise, and compound assignment operators,
# explaining their mathematical behavior and practical usage.
#
# KEY CONCEPTS USED:
# • `/` vs `//`: True division (`/`) always returns a `float`; floor division (`//`) truncates decimals.
# • `%` (Modulo): Computes division remainder, widely used in batch processing and round-robin scheduling.
# • `==` vs `=`: Single `=` assigns values; double `==` tests equality.
# • Logical Short-Circuiting: `and` stops evaluating as soon as one condition is False;
#   `or` stops evaluating as soon as one condition is True.
# • Compound Operators (`+=`, `-=`, `*=`): Cleaner, in-place variable updates.
#
# HOW IT CONNECTS TO THE PROJECT:
# Operators control loop boundaries, calculate token limits, compare distance scores
# (e.g. `if score < threshold:`), and manage retry attempts in agentic workflows.
#
# THINGS TO REMEMBER:
# • `not` has higher precedence than `and`, which has higher precedence than `or`.
#   Use parentheses `(A and B) or C` for clarity!
# • Division `/=` converts integers to floats automatically.
#
# COMMON MISTAKES TO AVOID:
# • Using a single `=` inside an `if` statement (causes a SyntaxError in Python).
# • Forgetting that `//` rounds down toward negative infinity (e.g. `-7 // 2` is `-4`, not `-3`).
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Compare `x / y` (float result) with `x // y` (truncated integer result).
# ============================================================
