"""
ReAct Agent — Raw Anthropic Tool Use
======================================
NO framework. Same concept as 01_raw_openai but using Anthropic's API.

KEY DIFFERENCES vs OpenAI:
  ┌─────────────────────────────────────────────────────────┐
  │ OpenAI                    │ Anthropic                   │
  ├───────────────────────────┼─────────────────────────────┤
  │ tools=[{type,function}]   │ tools=[{name,description,   │
  │                           │   input_schema}]            │
  ├───────────────────────────┼─────────────────────────────┤
  │ role: "tool"              │ role: "user" with           │
  │ tool_call_id: "..."       │   type:"tool_result"        │
  ├───────────────────────────┼─────────────────────────────┤
  │ assistant.tool_calls[]    │ content[] blocks with       │
  │                           │   type:"tool_use"           │
  ├───────────────────────────┼─────────────────────────────┤
  │ system in messages[]      │ system= top-level param     │
  └───────────────────────────┴─────────────────────────────┘

Anthropic's message format is more structured — each turn's content
is an array of typed blocks (text, tool_use, tool_result).
"""

import json
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.tools import execute_tool, anthropic_tool_schemas, TOOLS
from shared.display import step_think, step_action, step_observe, step_final, header, error

try:
    import anthropic as anthropic_sdk
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


SYSTEM_PROMPT = """You are a research assistant with access to tools.

Use the ReAct pattern: think, act (call a tool), observe the result, repeat.

For risky actions (save_file, send_email), explain your intent before calling.
Be concise but thorough. Cite sources from web_search / fetch_url results."""

MAX_ITERATIONS = 10


# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------

def run_agent(user_query: str, model: str = "claude-haiku-4-5-20251001", auto_confirm: bool = False) -> str:
    if not ANTHROPIC_AVAILABLE:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    client = anthropic_sdk.Anthropic(api_key=api_key)
    tool_schemas = anthropic_tool_schemas()

    # Anthropic: system is a top-level param, NOT in messages
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_query},
    ]

    header("Raw Anthropic Tool Use Agent")

    for iteration in range(MAX_ITERATIONS):
        # ── Call Claude ────────────────────────────────────────────────────
        response = client.messages.create(
            model=model,
            system=SYSTEM_PROMPT,   # <-- Anthropic-specific: system goes here
            max_tokens=4096,
            tools=tool_schemas,
            messages=messages,
        )

        # ── Collect tool_use blocks from the response ──────────────────────
        # response.content is a list of blocks: TextBlock | ToolUseBlock
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        # Show any text the model produced (its "thinking")
        for tb in text_blocks:
            if tb.text.strip():
                step_think(tb.text, framework="Anthropic")

        # ── No tool calls → final answer ───────────────────────────────────
        if not tool_use_blocks or response.stop_reason == "end_turn":
            final_text = " ".join(tb.text for tb in text_blocks if tb.text.strip())
            if not final_text:
                final_text = "(no response)"
            step_final(final_text, framework="Anthropic")
            return final_text

        # ── Has tool calls → append assistant turn, execute tools ──────────
        # Anthropic requires the full content array (text + tool_use blocks)
        messages.append({
            "role": "assistant",
            "content": [b.model_dump() for b in response.content],
        })

        # Build the tool_result blocks for the next user turn
        tool_results = []

        for block in tool_use_blocks:
            tool_name = block.name
            tool_use_id = block.id
            args = block.input  # already a dict for Anthropic (not JSON string)

            step_action(tool_name, args, framework="Anthropic")

            result, confirmed = execute_tool(tool_name, args, auto_confirm=auto_confirm)

            if not confirmed:
                result = f"User declined to run '{tool_name}'. Do not retry this action."

            step_observe(result, framework="Anthropic")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result,
            })

        # Anthropic: all tool results go in a SINGLE "user" turn as a list
        messages.append({
            "role": "user",
            "content": tool_results,
        })

    fallback = "Reached maximum iterations without a final answer."
    error(fallback, framework="Anthropic")
    return fallback


# ---------------------------------------------------------------------------
# Mock mode
# ---------------------------------------------------------------------------

def run_mock(user_query: str) -> str:
    header("Raw Anthropic Tool Use Agent (MOCK MODE)")
    print(f"Query: {user_query}\n")

    step_think("I'll search for this topic using web_search.", framework="Anthropic")
    step_action("web_search", {"query": user_query}, framework="Anthropic")
    result, _ = execute_tool("web_search", {"query": user_query}, auto_confirm=True)
    step_observe(result, framework="Anthropic")

    step_think("I'll fetch the most relevant URL for more detail.", framework="Anthropic")
    step_action("fetch_url", {"url": "https://climate.nasa.gov"}, framework="Anthropic")
    result2, _ = execute_tool("fetch_url", {"url": "https://climate.nasa.gov"}, auto_confirm=True)
    step_observe(result2, framework="Anthropic")

    answer = (
        f"Based on my research about '{user_query}':\n\n"
        f"Key findings from web search and URL retrieval.\n\n"
        f"[MOCK] In real mode this would be Claude's synthesized answer."
    )
    step_final(answer, framework="Anthropic")
    return answer


# ---------------------------------------------------------------------------
# Message structure explainer
# ---------------------------------------------------------------------------

def explain_message_flow():
    print("""
RAW ANTHROPIC MESSAGE STRUCTURE
─────────────────────────────────────────────────────────────────

Anthropic API call:
  client.messages.create(
    model="claude-...",
    system="You are a research assistant...",   # <-- top-level, not in messages
    max_tokens=4096,
    tools=[{"name": "web_search", "description": "...", "input_schema": {...}}],
    messages=[...]
  )

Turn 1 — USER:
  {"role": "user", "content": "What is climate change?"}

Turn 2 — ASSISTANT with tool_use blocks:
  {
    "role": "assistant",
    "content": [
      {"type": "text", "text": "Let me search for this."},
      {
        "type": "tool_use",
        "id": "toolu_abc123",
        "name": "web_search",
        "input": {"query": "climate change"}
      }
    ]
  }

Turn 3 — USER with tool_result blocks (ALL results in ONE turn):
  {
    "role": "user",
    "content": [
      {
        "type": "tool_result",
        "tool_use_id": "toolu_abc123",
        "content": "Search results: ..."
      }
    ]
  }

Turn 4 — ASSISTANT final answer:
  {"role": "assistant", "content": [{"type": "text", "text": "Climate change is..."}]}

─────────────────────────────────────────────────────────────────
""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    parser = argparse.ArgumentParser(description="Raw Anthropic ReAct Agent")
    parser.add_argument("query", nargs="?", default="Research Python async patterns and save a cheat-sheet to async_guide.txt")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = parser.parse_args()

    if args.explain:
        explain_message_flow()

    if args.mock or os.getenv("AGENT_MODE") == "mock":
        run_mock(args.query)
    else:
        run_agent(args.query, model=args.model)
