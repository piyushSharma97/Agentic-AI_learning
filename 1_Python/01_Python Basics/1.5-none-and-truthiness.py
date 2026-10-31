#!/usr/bin/env python
# coding: utf-8

"""
None and truthiness, and the JavaScript equivalent.

None means "no value". It is one object, so every None is the same None.

JavaScript comparison:
- Python None                     ->  JS null
- Python has no undefined.        ->  JS undefined means "not set"
- Python True / False             ->  JS true / false
- Python if items:                ->  JS if (items.length) for arrays,
                                      because an empty array is truthy in JS
                                      and falsy in Python

Falsy values in Python: None, False, 0, 0.0, "", [], {}, set()
Falsy values in JavaScript: false, 0, "", null, undefined, NaN
Empty arrays and empty objects are truthy in JavaScript. That is the trap.
"""

# --------------------------------------------------
# 1. None is the absence of a value
# --------------------------------------------------

result = None
print("Result:", result)
print("Is None?", result is None)

# A function with no return gives None.
# A JavaScript function with no return gives undefined.

def lookup(found):
    if found:
        return "document"
    return None


print("Missing lookup:", lookup(False))

# --------------------------------------------------
# 2. What counts as false in an if
# --------------------------------------------------

samples = [None, False, 0, 0.0, "", [], {}, set(), "hi", 3, [1]]

print("--- truthiness ---")
for value in samples:
    if value:
        print(repr(value), "is truthy")
    else:
        print(repr(value), "is falsy")

# --------------------------------------------------
# 3. Why this matters next to JavaScript
# --------------------------------------------------
# Python: an empty list skips the block.
# JavaScript: if ([]) { ... } RUNS, because [] is truthy.
# Check the length in JS. In Python, the value itself is enough.

chunks = []
if chunks:
    print("Have chunks")
else:
    print("No chunks yet — Python treats [] as false")

chunks.append("one paragraph")
if chunks:
    print("Now there is a chunk:", chunks[0])
