#!/usr/bin/env python
# coding: utf-8

"""
Modules and imports.

A module is any .py file. import loads that file once and gives you its names.
Packages are folders of modules. This lesson uses the two files sitting
next to it: math_utils.py and text_tools.py.

When you run this file, Python adds its folder to the search path,
so `import math_utils` finds math_utils.py in the same directory.

JavaScript comparison:
- Python import math_utils. JavaScript import * as mathUtils from "./math_utils.js".
- Python from text_tools import normalize. JavaScript import { normalize } from "./text_tools.js".
- Python import text_tools as tools. JavaScript import * as tools from "./text_tools.js".
- Python if __name__ == "__main__" runs only when the file is the program. Node uses import.meta.url compared with the process entry, or the older require.main === module.
"""

# --------------------------------------------------
# 1. Import the whole module
# --------------------------------------------------
# You then use module.function. The prefix makes the source of the name obvious.

import math_utils

score = 1.4
print("Clamped score:", math_utils.clamp(score, 0, 1))
print("800 tokens is about", math_utils.tokens_to_words(800), "words")

# --------------------------------------------------
# 2. Import specific names
# --------------------------------------------------
# After this, normalize and preview are local names. You do not write text_tools.

from text_tools import normalize, preview

query = "  What   Is   A Module?  "
print("Normalized query:", normalize(query))
print("Preview:", preview("Modules let you split a course into small files.", limit=24))

# --------------------------------------------------
# 3. Alias — a shorter name for a long module
# --------------------------------------------------

import text_tools as tools

print("Alias preview:", tools.preview("alias"))

# --------------------------------------------------
# 4. What import actually does
# --------------------------------------------------
# The module object has a __name__ (how it was imported)
# and a __file__ (the path of the .py file on disk).

print("Module name:", math_utils.__name__)
print("Module file:", math_utils.__file__)

# --------------------------------------------------
# 5. if __name__ == "__main__"
# --------------------------------------------------
# When you run a file directly, Python sets __name__ to "__main__".
# When another file imports it, __name__ is the module name.
# Put demos and script entry points under this check so importing
# the file does not start the demo.

def demo():
    print("This line runs because this file is the program being executed.")


if __name__ == "__main__":
    demo()
