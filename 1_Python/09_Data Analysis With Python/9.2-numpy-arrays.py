#!/usr/bin/env python
# coding: utf-8

"""
NumPy: arrays for numeric work.

A Python list can hold numbers, but each item is a full Python object.
A NumPy array stores numbers in one contiguous block, so math on a whole
column is one operation. Embeddings and score matrices are arrays.

JavaScript comparison:
- NumPy is not part of JavaScript. A plain array can hold the numbers, but scores + 5 loops in JS. NumPy adds 5 to every item in one expression.
- scores[scores >= 80] is scores.filter(score => score >= 80).
- A 2D array is an array of arrays. Shape and axis are NumPy ideas. JS reads embeddings[row][column].
- The dot product embeddings @ query is a small loop, or a library such as math.js. There is no @ operator in JavaScript.
"""

import numpy as np

# --------------------------------------------------
# 1. Build an array
# --------------------------------------------------

scores = np.array([88, 74, 93, 61, 90])
print("Array:", scores)
print("Type:", type(scores), "dtype:", scores.dtype, "shape:", scores.shape)

# --------------------------------------------------
# 2. Vectorised math — no Python loop
# --------------------------------------------------
# This adds 5 to every score. The list version would need a comprehension.

curved = scores + 5
print("Curved:", curved)
print("Mean:", scores.mean(), "Max:", scores.max(), "Min:", scores.min())

# --------------------------------------------------
# 3. Boolean mask — filter with a condition
# --------------------------------------------------
# scores >= 80 is an array of True/False. Using it as an index keeps the True rows.

passing = scores[scores >= 80]
print("Passing scores:", passing)

# --------------------------------------------------
# 4. Two dimensions — a tiny embedding matrix
# --------------------------------------------------
# Shape (3, 4) means 3 vectors, each with 4 numbers.
# axis=0 walks down rows. axis=1 walks across columns.

embeddings = np.array([
    [0.1, 0.2, 0.0, 0.4],
    [0.1, 0.0, 0.2, 0.3],
    [0.8, 0.1, 0.0, 0.0],
])
print("Shape:", embeddings.shape)
print("Mean of each vector:", embeddings.mean(axis=1))

# --------------------------------------------------
# 5. Dot product — a simple similarity score
# --------------------------------------------------
# Larger dot product means the two vectors point in a more similar direction.
# Real retrieval uses cosine similarity, which is a normalised dot product.

query = np.array([0.1, 0.2, 0.0, 0.5])
similarities = embeddings @ query
best = int(similarities.argmax())
print("Similarity to each row:", similarities)
print("Closest row index:", best, "vector:", embeddings[best])
