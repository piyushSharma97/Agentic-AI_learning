"""
Your first FastAPI app.

A web API is a program that waits for a request and sends back a response.
The request has three parts you will use in this file:

1. Method — what the caller wants to do. GET means "read". POST means "send data".
2. Path — the address, such as /health or /hello/Piyush.
3. Query — extra options after a ?, such as /find?topic=files.

FastAPI connects a URL to a Python function. The function's return value
is turned into JSON, which is text that looks like a Python dict.

How a call travels:
    browser or TestClient
        -> method + path  (GET /hello/Piyush)
        -> FastAPI finds the function decorated with that method and path
        -> FastAPI fills the function arguments from the URL
        -> your function returns a dict
        -> FastAPI turns that dict into JSON text and sends it back

This file only defines the app. It does not start a server.
12.1-first-api.py calls these routes and prints the answers.
To serve them for real, from this folder:

    uv run uvicorn first_api:app --reload

Then open http://127.0.0.1:8000/docs
/docs is a page FastAPI builds from these functions. You can try each route there.

JavaScript comparison:
- FastAPI() is express().
- @app.get("/health") is app.get("/health", handler).
- return {"status": "ok"} is res.json({ status: "ok" }). FastAPI turns the dict into JSON for you.
- /hello/{name} is Express /hello/:name.
"""

from fastapi import FastAPI

# FastAPI() builds the application object.
# title shows up on the /docs page. It does not change how the routes work.
# The name `app` matters: uvicorn first_api:app looks for a variable called app
# inside the file first_api.py.
app = FastAPI(title="Python lesson API")


# @app.get("/health") is a decorator.
# It registers health() as the function that runs for GET /health.
# Without the decorator, health() is a normal function nobody on the network can call.
@app.get("/health")
def health():
    """A tiny route used to ask "is the server alive?"."""
    # A dict becomes a JSON object. Keys become strings. This returns {"status": "ok"}.
    return {"status": "ok"}


@app.get("/topics")
def topics():
    """Return a list. JSON lists and Python lists look the same."""
    # The value can be a list, a number, or a nested dict. FastAPI serializes all of them.
    return {
        "count": 4,
        "topics": ["functions", "modules", "files", "classes"],
    }


# {name} is a path parameter. It is part of the address, not after the ?.
# GET /hello/Piyush sets name to the string "Piyush".
# The type hint str tells FastAPI the value is text. A path segment is always
# text on the wire. FastAPI converts it when you ask for int or float.
@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f"Hello, {name}."}


# topic and limit are query parameters because they are NOT in the path string.
# The caller writes /find?topic=files&limit=1
# FastAPI matches the argument name to the query name.
# limit: int = 2 means: convert the text to an integer, and use 2 if it was omitted.
@app.get("/find")
def find(topic: str, limit: int = 2):
    catalog = ["functions", "modules", "files", "classes", "fastapi"]
    matches = [item for item in catalog if topic.lower() in item]
    return {"topic": topic, "matches": matches[:limit]}
