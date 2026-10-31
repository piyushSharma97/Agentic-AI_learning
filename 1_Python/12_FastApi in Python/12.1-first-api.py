#!/usr/bin/env python
# coding: utf-8

"""
Call the routes in first_api.py and read the responses.

TestClient pretends to be a browser. It sends a request into the app
and gives you the response, without opening a port and without uvicorn.
That is why this file finishes and returns to the terminal.

A response has two parts you should look at every time:
- status_code: a number. 200 means the route ran and succeeded.
- json(): the dict FastAPI built from the function's return value.

To serve the same app for real, from this folder:

    uv run uvicorn first_api:app --reload

Then open http://127.0.0.1:8000/docs

JavaScript comparison:
- FastAPI is a Python web framework. Express is the usual JavaScript match.
- @app.get("/health") is app.get("/health", (req, res) => res.json({ status: "ok" })).
- Returning a dict becomes JSON. Express does that when you call res.json.
- FastAPI writes /docs for you. Express does not. People add Swagger as a separate package.
- TestClient calls the app without a port. Supertest does that for Express.
"""

from fastapi.testclient import TestClient

from first_api import app

# One client can call every route on this app.
# client.get(path) sends a GET request. Nothing is listening on the network.
client = TestClient(app)

# --------------------------------------------------
# 1. GET /health — no arguments
# --------------------------------------------------
# The path is the whole address. There is no {name} and no ?query.

health_response = client.get("/health")
print("GET /health")
print("  status:", health_response.status_code)  # 200 = OK
print("  body:", health_response.json())

# --------------------------------------------------
# 2. GET /topics — a nested dict and a list
# --------------------------------------------------

topics_response = client.get("/topics")
print("GET /topics")
print("  status:", topics_response.status_code)
print("  body:", topics_response.json())
print("  first topic:", topics_response.json()["topics"][0])

# --------------------------------------------------
# 3. GET /hello/{name} — a path parameter
# --------------------------------------------------
# The value is inside the path. /hello/Piyush fills name="Piyush".
# Change the last segment and the function receives a different string.

hello_response = client.get("/hello/Piyush")
print("GET /hello/Piyush")
print("  status:", hello_response.status_code)
print("  body:", hello_response.json())

# --------------------------------------------------
# 4. GET /find — query parameters
# --------------------------------------------------
# params is the dict TestClient puts after the ?.
# This request is GET /find?topic=file&limit=1
# topic is required. limit has a default, so it can be left out.

find_response = client.get("/find", params={"topic": "file", "limit": 1})
print("GET /find?topic=file&limit=1")
print("  status:", find_response.status_code)
print("  body:", find_response.json())

# Same route, default limit. Only topic is sent.
default_limit = client.get("/find", params={"topic": "s"})
print("GET /find?topic=s")
print("  status:", default_limit.status_code)
print("  body:", default_limit.json())

# A bad type does not enter find(). "lots" cannot become an int.
# FastAPI answers 422 itself. 422 means "I understood the URL, but the data is invalid."
bad_limit = client.get("/find", params={"topic": "files", "limit": "lots"})
print("GET /find?topic=files&limit=lots")
print("  status:", bad_limit.status_code)
print("  FastAPI rejected it before find() ran.")
