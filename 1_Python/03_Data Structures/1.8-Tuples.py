#!/usr/bin/env python
# coding: utf-8

"""
Tuples: an ordered collection you do not change.

A tuple uses parentheses. After it is created, you cannot add, remove,
or replace items. You can still read them by index, the same way as a list.

JavaScript comparison:
- There is no tuple type in JS.
- A JS array is the closest container, but it is mutable, like a Python list.
- Object.freeze([1, 2]) blocks some changes, but it is not a real tuple:
  nested objects inside it can still change.
- Python unpacking  a, b = pair
  JS destructuring   const [a, b] = pair
- A function can return a tuple to hand back several values.
  JS returns an array and the caller destructures it.
"""

# --------------------------------------------------
# 1. Creating a tuple
# --------------------------------------------------

empty_tuple = ()
print(empty_tuple)
print(type(empty_tuple))

print(type(list()))
print(type(tuple()))

mixed_tuple = (1, "Hello World", 3.14, True)
print(mixed_tuple)

# A one-item tuple needs a comma. (1) is just the number 1.
single = (1,)
print("One item:", single, type(single))

# --------------------------------------------------
# 2. Combine tuples — this builds a new tuple
# --------------------------------------------------

numbers = (10, 20, 30)
concatenation_tuple = numbers + mixed_tuple
print(concatenation_tuple)

# --------------------------------------------------
# 3. Read, do not write
# --------------------------------------------------

print("First number:", numbers[0])
print("Last number:", numbers[-1])
print("Slice:", numbers[0:2])

# numbers[0] = 99 would raise TypeError.
# A JS array would allow pair[0] = 99.

# --------------------------------------------------
# 4. Unpacking
# --------------------------------------------------
# The number of names must match the number of items.

x, y, z = numbers
print("Unpacked:", x, y, z)

# A function returning two results is the usual reason to use a tuple.

def min_max(values):
    return min(values), max(values)


low, high = min_max([4, 9, 1, 7])
print("Low:", low, "High:", high)
