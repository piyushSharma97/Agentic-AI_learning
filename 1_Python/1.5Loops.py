#!/usr/bin/env python
# coding: utf-8

"""
Python Iteration Masterclass: For Loops, While Loops & Control Flow

Why Loops are Fundamental in Generative AI:
1. Document Processing: Iterating over hundreds of document chunks to embed and store them.
2. Token Streaming: Iterating over live token streams emitted by LLMs (e.g. `for chunk in chain.stream()`).
3. Chatbot REPLs: An infinite `while True` loop that keeps terminal chatbots running until the user types 'exit'.
4. Multi-Agent Loops: Running agent reflection, tool execution, and retry cycles until a goal is achieved.

Loop Types:
• For Loop (Definite Iteration): Executes a fixed number of times over an iterable collection (list, string, range).
• While Loop (Indefinite Iteration): Repeatedly executes as long as a boolean condition remains True.
"""

# ==================================================
# 1. For Loop: Iterating Over a List
# ==================================================
# In each iteration, Python assigns the next item in the collection to the loop variable (`model`).

models = ["Claude-3.5-Sonnet", "GPT-4o", "Gemma-2-9B"]

print("--- 1. Iterating Over List ---")
for model in models:
    print("Model name: =", model)
    print("---")

# ==================================================
# 2. For Loop with range()
# ==================================================
# range(start, stop, step) generates an arithmetic progression of integers.
# IMPORTANT: The 'stop' value is EXCLUSIVE!
# range(1, 8) generates: 1, 2, 3, 4, 5, 6, 7 (does NOT include 8).

print("\n--- 2. Loop with range(1, 8) ---")
values = range(1, 8)
for i in values:
    print("value =", i)

# ==================================================
# 3. Iterating Through a String
# ==================================================
# Strings in Python are sequences of characters. A for loop visits each character one-by-one.

print("\n--- 3. Iterating Over String Characters ---")
greeting = "AI"
for char in greeting:
    print("Char:", char)

# ==================================================
# 4. Loop Control: 'break' Statement
# ==================================================
# 'break' terminates the loop IMMEDIATELY and jumps execution to the first statement outside the loop.
# Used when a search target has been found or an error condition occurs.

print("\n--- 4. The 'break' Statement (Exits at 3) ---")
for num in range(1, 11):
    if num == 3:
        print(f"Target {num} encountered! Breaking out of loop.")
        break
    print("Current num:", num)

# ==================================================
# 5. Loop Control: 'continue' Statement
# ==================================================
# 'continue' skips the REST of the current iteration and jumps directly to the NEXT iteration.
# It does NOT exit the loop.

print("\n--- 5. The 'continue' Statement (Skips 3) ---")
for num in range(1, 6):
    if num == 3:
        print(f"Skipping {num}...")
        continue
    print("Processed:", num)

# ==================================================
# 6. The Unique Python 'for...else' Pattern
# ==================================================
# In Python, a loop can have an 'else' block!
# RULE: The 'else' block executes ONLY IF the loop finishes normally WITHOUT hitting a 'break'!
# If the loop exits early via 'break', the 'else' block is SKIPPED.
# This eliminates the need for manual boolean flags like `found = False`.

print("\n--- 6. The 'for...else' Search Pattern ---")
stock = ['Laptop', 'Keyboard', 'Mouse']
order = "Keyboard"  # Try changing this to "Headphones" to see the else block trigger!

for product in stock:
    if product == order:
        print(f"Found: {order} is in stock. Adding to cart.")
        break  # Loop breaks here -> 'else' will NOT run!
else:
    # Runs only if the loop completed without encountering break
    print(f"Search complete: Sorry, {order} is out of stock.")

# ==================================================
# 7. Discard Variable Pattern (_)
# ==================================================
# By Python convention, an underscore (_) is used when you need to repeat an action N times
# but don't care about the index value itself.

print("\n--- 7. Discard Variable Loop (_) ---")
for _ in range(3):
    print("Ping local LLM server...")

# ==================================================
# 8. Nested For Loops
# ==================================================
# An inner loop runs to completion for every single iteration of the outer loop.
# Total iterations = len(attributes) * len(cars) = 2 * 3 = 6 iterations.

print("\n--- 8. Nested Loops ---")
attributes = ['Electric', 'Autonomous']
cars = ['Tesla', 'Waymo', 'Cruise']

for att in attributes:
    for car in cars:
        print(f"{att} -> {car}")

# ==================================================
# 9. While Loops (Indefinite Iteration)
# ==================================================
# Repeats as long as the condition evaluates to True.
# Always ensure the loop body modifies a variable towards False to avoid infinite loops!

print("\n--- 9. Countdown While Loop ---")
n = 5
while n > 0:
    print(f"Countdown: {n}")
    n -= 1  # Decrement n so loop eventually terminates!
print("Blastoff!")

# In conversational chatbots, while loops power the interactive session:
# while True:
#     user_text = input("You: ")
#     if user_text == "exit": break
#     response = bot.invoke(...)

# ============================================================
# REVISION NOTES — 1.5Loops.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script is a comprehensive guide to iteration in Python, demonstrating `for` loops, `while` loops,
# numeric generation via `range()`, loop control (`break`, `continue`), throwaway index variables (`_`),
# nested loops, and the Python-specific `for...else` pattern.
#
# KEY CONCEPTS USED:
# • For Loop: Iterates over iterable sequences (lists, tuples, ranges, strings, dictionaries).
# • While Loop: Repeats as long as a condition is True; requires state updates to avoid infinite loops.
# • `break`: Abruptly exits the loop immediately.
# • `continue`: Skips the remainder of the current pass and advances to the next item.
# • `for...else`: Runs only when the loop completes naturally without hitting a `break`.
#
# HOW IT CONNECTS TO THE PROJECT:
# Loops are ubiquitous in AI applications:
# - Streaming response tokens: `for chunk in chain.stream(): ...`
# - Indexing documents: `for doc in documents: ...`
# - Chatbot prompt-response loops: `while True: user_input = input(...)`
#
# THINGS TO REMEMBER:
# • `range(start, stop)` does not include `stop`.
# • Avoid modifying a list while iterating over it; iterate over a copy instead (`for item in my_list[:]:`).
#
# COMMON MISTAKES TO AVOID:
# • Forgetting the loop variable decrement/increment in a `while` loop, causing an infinite freeze.
# • Confusing `break` (terminates loop entirely) with `continue` (only skips current turn).
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Compare the `break` example in Section 4 with the `continue` example in Section 5.
# ============================================================
