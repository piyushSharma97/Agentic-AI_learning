#!/usr/bin/env python
# coding: utf-8

"""
Special methods (dunder methods).

Python looks for methods with double underscores when you use built-in
syntax. Defining them lets your objects work with print, len, ==, and +.

You already rely on these when you print a LangChain message or compare models.

JavaScript comparison:
- __repr__ and __str__ are close to toString(). JavaScript does not have two separate hooks. String(obj) and template interpolation both call toString.
- __len__ is what len() calls. JavaScript uses a .length property instead of a method the language calls for you.
- __eq__ is what == calls. JavaScript === on objects compares references, not your own rule, unless you write a method and call it yourself.
- __add__ is what + calls. JavaScript + on objects becomes string concatenation unless the value is a number or bigint.
"""

class Vector:
    """A tiny stand-in for an embedding: a list of numbers with a few operations."""

    def __init__(self, values):
        self.values = list(values)

    def __repr__(self):
        # repr is the unambiguous text shown in the shell and in logs.
        return f"Vector({self.values})"

    def __len__(self):
        # len(vector) calls this.
        return len(self.values)

    def __eq__(self, other):
        # vector_a == vector_b compares values, not object identity.
        if not isinstance(other, Vector):
            return False
        return self.values == other.values

    def __add__(self, other):
        # vector_a + vector_b builds a new Vector. It does not change either input.
        if len(self) != len(other):
            raise ValueError("Vectors must be the same length")
        added = [a + b for a, b in zip(self.values, other.values)]
        return Vector(added)


query = Vector([1, 2, 3])
other = Vector([1, 2, 3])
shift = Vector([0.5, 0, 1])

print(query)                  # uses __repr__
print("Dimensions:", len(query))
print("Equal values?", query == other)
print("Same object?", query is other)
print("Added:", query + shift)

# --------------------------------------------------
# __str__ vs __repr__
# --------------------------------------------------
# print() prefers __str__ when it exists, and falls back to __repr__.
# Keep __repr__ useful for debugging. Use __str__ for a friendly sentence.

class Lesson:
    def __init__(self, title):
        self.title = title

    def __str__(self):
        return f"Lesson: {self.title}"

    def __repr__(self):
        return f"Lesson(title={self.title!r})"


current = Lesson("special methods")
print(current)
print(repr(current))
