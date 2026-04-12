"""
Shared tool implementations used across all four agent frameworks.

Each tool is a plain Python function. Frameworks wrap them differently,
but the underlying logic is identical — making cross-framework comparison fair.

RISKY TOOLS (require human confirmation before execution):
  - save_file: writes to disk
  - send_email: sends an email

SAFE TOOLS (run automatically):
  - web_search: search the web (mocked)
  - fetch_url: retrieve a URL (mocked)
  - calculate: evaluate a math expression
"""

import math
import os
import json
import time
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Mock data — swap these out for real API calls (Tavily, httpx, etc.)
# ---------------------------------------------------------------------------

MOCK_SEARCH_DB = {
    "climate change": [
        {"title": "NASA Climate Change Evidence", "url": "https://climate.nasa.gov", "snippet": "97% of climate scientists agree climate change is real and human-caused. Global temperature has risen ~1.1°C since pre-industrial times."},
        {"title": "IPCC Sixth Assessment Report", "url": "https://www.ipcc.ch/ar6", "snippet": "Without immediate action, warming will exceed 1.5°C by 2030s. Sea level rise, extreme weather, and biodiversity loss are accelerating."},
    ],
    "python async": [
        {"title": "Python asyncio docs", "url": "https://docs.python.org/3/library/asyncio.html", "snippet": "asyncio is a library to write concurrent code using the async/await syntax. Best for I/O-bound and high-level structured network code."},
        {"title": "Real Python: Async IO", "url": "https://realpython.com/async-io-python", "snippet": "async def defines a coroutine. await suspends execution until the awaited thing completes. The event loop runs coroutines."},
    ],
    "react hooks": [
        {"title": "React Hooks Overview", "url": "https://react.dev/reference/react", "snippet": "Hooks let you use state and lifecycle features in function components. useState, useEffect, useCallback, useMemo are the most common."},
        {"title": "Rules of Hooks", "url": "https://react.dev/warnings/invalid-hook-call-warning", "snippet": "Only call hooks at the top level, not inside loops or conditions. Only call hooks from React function components."},
    ],
}

MOCK_URL_DB = {
    "https://climate.nasa.gov": "NASA CLIMATE CHANGE\n\nKey Facts:\n- CO2 levels: 421 ppm (highest in 800,000 years)\n- Global temperature: +1.1°C since 1880\n- Arctic ice: declining 13% per decade\n- Sea level: rising 3.7mm/year\n\nProjections: If emissions continue, 2-4°C warming by 2100.",
    "https://docs.python.org/3/library/asyncio.html": "asyncio — Asynchronous I/O\n\nasyncio is used as a foundation for multiple Python asynchronous frameworks.\nProvides: event loop, coroutines, tasks, futures, streams, synchronization primitives.\n\nExample:\n  async def main():\n      await asyncio.sleep(1)\n  asyncio.run(main())",
}


# ---------------------------------------------------------------------------
# Safe tools
# ---------------------------------------------------------------------------

def web_search(query: str) -> str:
    """
    Search the web for information.
    Returns top results with titles, URLs, and snippets.
    """
    time.sleep(0.1)  # simulate network latency

    # find best match in mock DB
    query_lower = query.lower()
    results = []
    for key, hits in MOCK_SEARCH_DB.items():
        if any(word in query_lower for word in key.split()):
            results.extend(hits)

    if not results:
        results = [{"title": f"Search results for '{query}'", "url": "https://example.com", "snippet": f"Found general information about {query}. This is a mock result — connect a real search API for live data."}]

    output = f"Search results for: '{query}'\n\n"
    for i, r in enumerate(results[:3], 1):
        output += f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}\n\n"
    return output.strip()


def fetch_url(url: str) -> str:
    """
    Retrieve the content of a URL.
    Returns the page text.
    """
    time.sleep(0.1)  # simulate network latency

    if url in MOCK_URL_DB:
        return MOCK_URL_DB[url]

    return f"[Fetched {url}]\nMock content: This page discusses topics related to {url.split('/')[-1].replace('-', ' ')}. For real content, replace this mock with an httpx/requests call."


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Supports: +, -, *, /, **, sqrt, sin, cos, log, etc.
    """
    # safe eval: only allow math operations
    allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return f"Result: {result}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# ---------------------------------------------------------------------------
# Risky tools — these MUST go through human confirmation before running
# ---------------------------------------------------------------------------

def save_file(filename: str, content: str) -> str:
    """
    [RISKY] Save content to a file on disk.
    Requires human confirmation before execution.
    """
    # Sanitize path — only allow writes inside ./agent_output/
    safe_dir = os.path.join(os.path.dirname(__file__), "..", "agent_output")
    os.makedirs(safe_dir, exist_ok=True)
    safe_path = os.path.join(safe_dir, os.path.basename(filename))

    with open(safe_path, "w") as f:
        f.write(content)

    return f"File saved: {safe_path} ({len(content)} bytes)"


def send_email(to: str, subject: str, body: str) -> str:
    """
    [RISKY] Send an email.
    Requires human confirmation before execution.
    This is mocked — no real email is sent.
    """
    timestamp = datetime.now().isoformat()
    return f"[MOCK] Email sent at {timestamp}\n  To: {to}\n  Subject: {subject}\n  Body preview: {body[:100]}..."


# ---------------------------------------------------------------------------
# Tool registry — maps name → (function, is_risky, description, parameters)
# ---------------------------------------------------------------------------

TOOLS: dict[str, dict[str, Any]] = {
    "web_search": {
        "fn": web_search,
        "risky": False,
        "description": "Search the web for information on any topic. Returns titles, URLs, and snippets.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"],
        },
    },
    "fetch_url": {
        "fn": fetch_url,
        "risky": False,
        "description": "Retrieve and read the full content of a URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"}
            },
            "required": ["url"],
        },
    },
    "calculate": {
        "fn": calculate,
        "risky": False,
        "description": "Evaluate a mathematical expression. Supports arithmetic, sqrt, sin, cos, log, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression, e.g. 'sqrt(144) + 2**10'"}
            },
            "required": ["expression"],
        },
    },
    "save_file": {
        "fn": save_file,
        "risky": True,  # <-- triggers human confirmation
        "description": "[RISKY] Save text content to a file on disk. Requires user confirmation.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Filename to save (e.g. 'report.txt')"},
                "content": {"type": "string", "description": "Text content to write"},
            },
            "required": ["filename", "content"],
        },
    },
    "send_email": {
        "fn": send_email,
        "risky": True,  # <-- triggers human confirmation
        "description": "[RISKY] Send an email to a recipient. Requires user confirmation.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject line"},
                "body": {"type": "string", "description": "Email body text"},
            },
            "required": ["to", "subject", "body"],
        },
    },
}


def execute_tool(name: str, args: dict[str, Any], auto_confirm: bool = False) -> tuple[str, bool]:
    """
    Execute a tool by name with given args.

    For risky tools, prompts the user for confirmation unless auto_confirm=True.

    Returns:
        (result_string, was_confirmed)
        was_confirmed=False means user denied the action.
    """
    if name not in TOOLS:
        return f"Unknown tool: {name}", True

    tool = TOOLS[name]

    if tool["risky"] and not auto_confirm:
        confirmed = _ask_human_confirmation(name, args)
        if not confirmed:
            return f"Action '{name}' was cancelled by the user.", False

    try:
        result = tool["fn"](**args)
        return result, True
    except Exception as e:
        return f"Tool error ({name}): {e}", True


def _ask_human_confirmation(tool_name: str, args: dict[str, Any]) -> bool:
    """Pause execution and ask the human to approve a risky action."""
    print("\n" + "=" * 60)
    print("  HUMAN CONFIRMATION REQUIRED")
    print("=" * 60)
    print(f"  Tool    : {tool_name}")
    print(f"  Args    : {json.dumps(args, indent=4)}")
    print("=" * 60)
    response = input("  Allow this action? [y/N]: ").strip().lower()
    print("=" * 60 + "\n")
    return response in ("y", "yes")


# ---------------------------------------------------------------------------
# OpenAI-format tool schemas (used by raw OpenAI + OpenAI Agents SDK)
# ---------------------------------------------------------------------------

def openai_tool_schemas() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": info["description"],
                "parameters": info["parameters"],
            },
        }
        for name, info in TOOLS.items()
    ]


# ---------------------------------------------------------------------------
# Anthropic-format tool schemas (used by raw Anthropic agent)
# ---------------------------------------------------------------------------

def anthropic_tool_schemas() -> list[dict]:
    return [
        {
            "name": name,
            "description": info["description"],
            "input_schema": info["parameters"],
        }
        for name, info in TOOLS.items()
    ]
