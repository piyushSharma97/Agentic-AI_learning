# Topic 10: Search Tools (Arxiv, Wikipedia, Tavily)

Welcome to **Topic 10** of the LangGraph Components course. This module covers testing external research and search tools **on their own** before plugging them into an agent.

---

## 🎯 Learning Objectives

By the end of this module, you will understand:
1. Why you should **always test tools independently** before attaching them to an LLM.
2. How to build an arXiv research tool using the modern `arxiv.Client` API (which replaces legacy deprecated wrappers).
3. How to query encyclopedic knowledge with the Wikipedia API wrapper.
4. How to execute real-time web searches using the Tavily Search API.
5. Why output truncation (e.g. max 500 characters) is critical to prevent tool observations from overwhelming the LLM context window.

---

## 🧠 Mental Model: The Specialist Research Suite

Different questions require different research sources:

```
┌─────────────────────────────────────────────────────────────┐
│                    SPECIALIST SEARCH TOOLS                  │
├───────────────────┬───────────────────┬─────────────────────┤
│ arXiv             │ Wikipedia         │ Tavily              │
├───────────────────┼───────────────────┼─────────────────────┤
│ • Research papers │ • Broad overviews │ • Today's news      │
│ • Paper IDs       │ • Historical facts│ • Live web prices   │
│   (1706.03762)    │ • Definitions     │ • Breaking events   │
│ • Academic peers  │ • Pop culture     │ • Live docs         │
│ • NO API key      │ • NO API key      │ • Needs API Key     │
└───────────────────┴───────────────────┴─────────────────────┘
```

If an agent gives a hallucinated or broken response, the problem is either:
1. The tool returned garbage or failed.
2. The model summarized the tool output poorly.

Testing the tools on their own isolates tool failure from model reasoning failure!

---

## 💻 Code Walkthrough

### 1. Modern arXiv Tool
```python
import arxiv
from langchain_core.tools import tool

@tool
def search_arxiv(query: str) -> str:
    """Search arXiv by keywords or paper ID."""
    client = arxiv.Client(page_size=2)
    # Search by ID or query...
    papers = list(client.results(search))
    return formatted_papers[:500]
```
> [!NOTE]
> The legacy `ArxivQueryRun` from `langchain_community` breaks on modern `arxiv >= 4.0` because `Search.results()` was removed. Using `arxiv.Client` directly guarantees long-term stability!

### 2. Wikipedia Tool
```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Limit to 500 chars to save LLM tokens!
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
print(wiki.invoke("Machine learning"))
```

### 3. Tavily Live Web Search
```python
from langchain_tavily import TavilySearch

# Requires TAVILY_API_KEY in .env
tavily = TavilySearch(max_results=3)
result = tavily.invoke({"query": "Latest AI news"})
```

---

## 🚀 How to Run

From the project root:
```powershell
python 10_LangGrapph_Component/10_search_tools/main.py
```

Or from inside this directory:
```powershell
cd 10_LangGrapph_Component/10_search_tools
python main.py
```

---

## 📝 Summary Checklist

| Tool | Best For | Requires Key? |
|---|---|---|
| **`search_arxiv`** | Academic research, ML papers, formulas | ❌ No |
| **`Wikipedia`** | Encyclopedic summaries, definitions | ❌ No |
| **`Tavily`** | Live real-time internet search |  Yes (`TAVILY_API_KEY`) |
