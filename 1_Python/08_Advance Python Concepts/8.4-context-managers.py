#!/usr/bin/env python
# coding: utf-8

"""
Context managers: code that runs on the way in and on the way out.

with open(...) is the one you already use. The file closes when the block
ends, including when an exception is raised.

You can write your own with a class that has __enter__ and __exit__,
or with the contextlib helper shown at the bottom.

JavaScript comparison:
- Python with timer: is the block that guarantees cleanup. JavaScript uses try / finally for the same guarantee.
- Newer JavaScript has the using keyword for disposable resources. It is the closest syntax to with.
- __enter__ is the setup. __exit__ is the cleanup in finally.
- @contextmanager is a small helper around a generator. JavaScript would write the try / finally by hand.
"""

from contextlib import contextmanager
import time

# --------------------------------------------------
# 1. Class-based context manager
# --------------------------------------------------

class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.start = time.perf_counter()
        print(f"{self.label}: start")
        return self  # the name after `as` receives this

    def __exit__(self, exc_type, exc, traceback):
        elapsed = time.perf_counter() - self.start
        print(f"{self.label}: {elapsed:.6f}s")
        return False  # False means: do not swallow the exception


with Timer("word count"):
    words = "context managers clean up after a block".split()
    print("Count:", len(words))

# --------------------------------------------------
# 2. __exit__ still runs when the block fails
# --------------------------------------------------

try:
    with Timer("failing block"):
        raise RuntimeError("boom")
except RuntimeError as error:
    print("Outside the with block, the error is still visible:", error)

# --------------------------------------------------
# 3. @contextmanager — the same idea as a generator
# --------------------------------------------------
# Code before yield is __enter__. Code after yield is __exit__.

@contextmanager
def section(title):
    print(f"----- {title} -----")
    yield
    print(f"----- end {title} -----")


with section("lesson"):
    print("This line sits inside the section.")
