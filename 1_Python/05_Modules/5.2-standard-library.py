#!/usr/bin/env python
# coding: utf-8

"""
The standard library: modules that ship with Python.

You do not install these. import is enough.
A few that show up constantly in AI projects:
- pathlib for file paths
- json for message payloads
- datetime for timestamps
- math for numeric helpers

JavaScript comparison:
- pathlib.Path is the path module: import path from "node:path".
- json.dumps is JSON.stringify. json.loads is JSON.parse.
- datetime.now() is new Date(). Python's timezone.utc is close to date.toISOString().
- math.sqrt and math.floor are Math.sqrt and Math.floor.
- These Python modules ship with the language. Node ships path, and JSON is global. A browser page has JSON and Math, not pathlib.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------
# 1. pathlib — paths as objects, not raw strings
# --------------------------------------------------
# Path(__file__) is this lesson file. .parent is the folder it lives in.

here = Path(__file__).resolve().parent
print("This lesson lives in:", here.name)
print("Project folder above it:", here.parent.name)

# --------------------------------------------------
# 2. json — text form of dicts and lists
# --------------------------------------------------
# Model APIs and tool results travel as JSON text.
# dumps = Python object to text. loads = text back to a Python object.

payload = {"role": "user", "content": "What is a module?"}
as_text = json.dumps(payload)
print("JSON text:", as_text)
print("Parsed again:", json.loads(as_text)["role"])

# --------------------------------------------------
# 3. datetime — aware timestamps
# --------------------------------------------------
# timezone.utc avoids a naive local time that is hard to compare later.

now = datetime.now(timezone.utc)
print("UTC time:", now.strftime("%Y-%m-%d %H:%M:%S"))

# --------------------------------------------------
# 4. math — numeric helpers
# --------------------------------------------------
print("Square root of 144:", math.sqrt(144))
print("Floor of 3.9:", math.floor(3.9))
