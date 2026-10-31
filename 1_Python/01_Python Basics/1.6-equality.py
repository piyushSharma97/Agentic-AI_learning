#!/usr/bin/env python
# coding: utf-8

"""
Equality and identity, and the JavaScript equivalent.

== asks "same value?"
is asks "same object in memory?"

JavaScript comparison:
- Python ==                        ->  JS ===   (compare value, no type coercion)
- Python is                       ->  JS ===   when both sides are objects
                                      (same reference)
- JS == is loose and coerces types. Python has no matching operator.
  JS "5" == 5 is true. Python "5" == 5 is false.
"""

# --------------------------------------------------
# 1. == compares values
# --------------------------------------------------

print("5 == 5:", 5 == 5)
print('"5" == 5:', "5" == 5)  # False. JS == would say true.

one = 1
print("True == 1:", True == one)  # True, because bool is a kind of int
print("True is 1:", True is one)  # False. Same value, not the same object.

# --------------------------------------------------
# 2. is compares identity
# --------------------------------------------------
# Two equal lists are not the same list.

first = [1, 2]
second = [1, 2]
third = first

print("first == second:", first == second)
print("first is second:", first is second)
print("first is third:", first is third)

# --------------------------------------------------
# 3. Use is for None, == for values
# --------------------------------------------------
# None is a single object, so `is None` is the normal check.
# JavaScript uses === null, and a separate check for undefined.

score = None
print("score is None:", score is None)
print("score == None:", score == None)  # works, but `is` is the form to use
