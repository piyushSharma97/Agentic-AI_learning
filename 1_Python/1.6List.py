#!/usr/bin/env python
# coding: utf-8

"""
Python Lists Masterclass: Creation, Indexing, Slicing, Modification & Methods

What is a List in Python?
A list is an ordered, mutable (changeable) collection of items enclosed in square brackets `[ ]`.
Lists are one of the most heavily used data structures in Python and AI engineering:
- Lists of messages: `[SystemMessage(...), HumanMessage(...), AIMessage(...)]`
- Lists of documents: `[Document(page_content="..."), ...]`
- Embedding vectors: `[0.012, -0.045, 0.981, ...]`

Key Properties of Lists:
1. Ordered: Items maintain their exact position; you can access items by index.
2. Mutable: You can add, remove, and change items after the list is created without creating a new object.
3. Heterogeneous: A single list can contain mixed data types (integers, strings, floats, objects).
4. Allows Duplicates: Multiple identical items can coexist within the same list.
"""

# ==================================================
# 1. Creating Lists
# ==================================================

# Homogeneous list (all strings)
cart = ["T-shirt", "Lamp", "Pen"]
print("Initial shopping cart:", cart)

# Heterogeneous list (mixed data types)
mixed_list = [1, "Python", 3.14, True]
print("Mixed list:", mixed_list)

# Empty list initialization (often used to collect chunks or messages)
empty_list = []
print("Empty list:", empty_list, "Type:", type(empty_list))

# ==================================================
# 2. Positive & Negative Indexing
# ==================================================
# Positive Indexing: Zero-based, counts from left to right (0, 1, 2, ...).
# Negative Indexing: Counts backward from the end (-1 is last item, -2 is second to last).

languages = ["Python", "Swift", "C++", "Rust"]

# Positive indexing
print("\n--- Indexing ---")
print(f"First item (index 0): {languages[0]}")
print(f"Third item (index 2): {languages[2]}")

# Negative indexing
print(f"Last item (index -1): {languages[-1]}")
print(f"Second to last (index -2): {languages[-2]}")

# ==================================================
# 3. Updating and Adding Items (append, extend, insert)
# ==================================================
# Lists are mutable, allowing in-place modification.

cart = ["T-shirt", "Lamp", "Pen"]

# 3.1 Update item at index 1 in place:
cart[1] = "Shoes"
print("\nAfter updating index 1:", cart)  # ['T-shirt', 'Shoes', 'Pen']

# 3.2 append(item): Appends ONE item to the very end of the list
cart.append("Book")
print("After append('Book'):", cart)

# 3.3 extend(iterable): Iterates through another collection and appends each element individually
# (Crucial difference: append([a, b]) would add a nested list; extend adds them as individual elements!)
fav_items = ["Headphones", "Phone"]
cart.extend(fav_items)
print("After extend(fav_items):", cart)

# 3.4 insert(index, item): Inserts an item at a specific index, shifting subsequent items to the right
cart.insert(2, "Smartwatch")
print("After insert(2, 'Smartwatch'):", cart)

# ==================================================
# 4. Removing Items (remove, pop, clear, del)
# ==================================================

# 4.1 remove(value): Removes the FIRST occurrence of the specified value.
# (Raises ValueError if the item is not found).
cart.remove("Pen")
print("\nAfter remove('Pen'):", cart)

# 4.2 pop(index=-1): Removes AND returns the item at the given index (defaults to the last item).
# This allows using lists as LIFO (Last-In-First-Out) stacks!
popped_item = cart.pop()
print(f"Popped item: '{popped_item}'")
print("Cart after pop():", cart)

# 4.3 del statement: Deletes an item at a specific index or deletes the entire variable from memory.
del cart[1]
print("After del cart[1]:", cart)

# 4.4 clear(): Empties the list completely, leaving an empty list `[]`.
cart.clear()
print("After clear():", cart)

# ==================================================
# 5. List Slicing: [start:stop:step]
# ==================================================
# Slicing extracts a sub-list:
# • start: Starting index (inclusive). Defaults to 0.
# • stop: Ending index (EXCLUSIVE!). Defaults to len(list).
# • step: Stride/interval. Defaults to 1.

fruits = ["apple", "banana", "mango", "cherry", "papaya", "watermelon", "orange"]

print("\n--- Slicing Fruits ---")
print("fruits[2:4] (index 2 and 3):", fruits[2:4])       # ['mango', 'cherry']
print("fruits[3:] (from index 3 to end):", fruits[3:])    # ['cherry', 'papaya', 'watermelon', 'orange']
print("fruits[:3] (first 3 items):", fruits[:3])          # ['apple', 'banana', 'mango']
print("fruits[::2] (every 2nd item):", fruits[::2])       # ['apple', 'mango', 'papaya', 'orange']
print("fruits[::-1] (reversed list):", fruits[::-1])      # Reverses list via negative step!

# Slice assignment replaces a slice with new items:
fruits[2:4] = ["kiwi", "grape"]
print("After slice assignment fruits[2:4] = ['kiwi', 'grape']:\n ", fruits)

# ==================================================
# 6. Common List Methods: count() and reverse()
# ==================================================

# count(item): Returns total occurrences of a value
fruits.insert(1, "apple")
print("\nTotal 'apple' count:", fruits.count("apple"))

# CRITICAL GOTCHA: reverse() mutates the list IN-PLACE and returns None!
# Do NOT do: `reversed_list = fruits.reverse()` -> that sets `reversed_list` to None!
print("fruits.reverse() return value:", fruits.reverse())  # Returns None!
print("Actual list reversed in-place:", fruits)

# ============================================================
# REVISION NOTES — 1.6List.py
# ============================================================
#
# WHAT THIS FILE DOES:
# This script covers Python lists from basic initialization and zero-based/negative indexing
# to mutation methods (`append`, `extend`, `insert`, `remove`, `pop`), slicing syntax,
# and in-place methods (`reverse`, `count`).
#
# KEY CONCEPTS USED:
# • Mutable Sequence: Lists can be altered in-place after creation without changing object identity.
# • Zero-Based & Negative Indexing: Index `0` is first, `-1` is last.
# • Slicing `[start:stop]`: Extracts a sublist; `stop` is always exclusive.
# • `append` vs `extend`:
#   - `append(x)`: Adds `x` as a single element.
#   - `extend(iter)`: Unpacks items from `iter` and adds each individually.
# • In-Place Mutation: Methods like `reverse()` and `sort()` modify the existing list and return `None`.
#
# HOW IT CONNECTS TO THE PROJECT:
# Lists represent nearly all multi-item collections in the AI codebase:
# - Storing dialogue turns: `messages = [HumanMessage(...), AIMessage(...)]`
# - Holding text chunks: `chunks = text_splitter.split_documents(docs)`
# - Embedding vectors: `vector = [0.12, -0.04, 0.88, ...]`
#
# THINGS TO REMEMBER:
# • Use `list.pop()` to retrieve and remove the most recent item (stack behavior).
# • Slicing `my_list[:]` creates a shallow copy of the list, preventing unexpected mutations.
#
# COMMON MISTAKES TO AVOID:
# • IndexError: Accessing an index outside range (e.g. `list[10]` on a 3-item list).
# • Assigning the result of `list.reverse()` or `list.sort()` to a variable (which evaluates to `None`).
# • Confusing `append` with `extend` when combining two lists.
#
# IF YOU FORGET HOW THIS WORKS, START HERE:
# Look at Section 3 (`append`, `extend`, `insert`) and Section 5 (`[start:stop]` slicing).
# ============================================================
