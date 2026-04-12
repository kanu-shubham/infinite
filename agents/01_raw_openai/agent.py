"""
ReAct Agent — Raw OpenAI Function Calling
==========================================
NO framework. We hand-roll the entire loop.

The ReAct loop:
  1. Send messages to GPT with tool schemas
  2. If response has tool_calls → execute tools, append results, go to 1
  3. If response is text only → we're done (final answer)

Key concepts demonstrated here:
  - Tool schemas as JSON (openai_tool_schemas)
  - message roles: system, user, assistant, tool
  - tool_call_id linking calls to results
  - Risky action interception BEFORE tool execution
  - Max iteration guard (prevents infinite loops)
  - Error recovery: bad tool args → inject error as tool result, keep going
"""

import json
import os
import sys
from typing import Any

# allow importing from sibling 'shared' package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.tools import execute_tool, openai_tool_schemas, TOOLS
from shared.display import step_think, step_action, step_observe, step_final, header, error

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a research assistant with access to tools.

Use the ReAct pattern:
  1. Think about what you need to find out
  2. Call a tool to gather information
  3. Observe the result
  4. Repeat until you have enough to answer

For risky actions (save_file, send_email), always explain what you plan to do
BEFORE calling the tool, so the user understands what will happen.

Be concise but thorough. Cite sources when you use web_search or fetch_url."""

MAX_ITERATIONS = 10  # safety limit


# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------

def run_agent(user_query: str, model: str = "gpt-4o-mini", auto_confirm: bool = False) -> str:
    """
    Run the ReAct agent on user_query.

    Returns the final answer string.
    Raises if OpenAI is not available.
    """
    if not OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    client = OpenAI(api_key=api_key)
    tool_schemas = openai_tool_schemas()

    # Message history — this is the agent's "memory" / context
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    header("Raw OpenAI ReAct Agent")

    for iteration in range(MAX_ITERATIONS):
        # ── Step 1: Call the LLM ──────────────────────────────────────────
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas,
            tool_choice="auto",  # let the model decide when to use tools
        )

        message = response.choices[0].message

        # ── Step 2: No tool calls → final answer ─────────────────────────
        if not message.tool_calls:
            final = message.content or "(no response)"
            step_final(final, framework="OpenAI")
            return final

        # ── Step 3: Has tool calls → show thinking + execute each tool ───
        # The model sometimes puts reasoning in content before tool calls
        if message.content:
            step_think(message.content, framework="OpenAI")

        # Append assistant message (with tool_calls) to history
        messages.append(message)

        # Execute every tool call in this turn
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_call_id = tool_call.id

            # Parse args — catch malformed JSON from the model
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                err_msg = f"Invalid JSON in tool args: {e}"
                error(err_msg, framework="OpenAI")
                # Inject the error as a tool result so the model can recover
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": err_msg,
                })
                continue

            step_action(tool_name, args, framework="OpenAI")

            # Execute — risky tools pause for confirmation here
            result, confirmed = execute_tool(tool_name, args, auto_confirm=auto_confirm)

            if not confirmed:
                result = f"User declined to run '{tool_name}'. Do not retry this action."

            step_observe(result, framework="OpenAI")

            # Append tool result so the model sees what happened
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result,
            })

    # If we hit MAX_ITERATIONS without a final answer, return what we have
    fallback = "Reached maximum iterations without a final answer."
    error(fallback, framework="OpenAI")
    return fallback


# ---------------------------------------------------------------------------
# Mock mode — runs without API key, shows the loop structure
# ---------------------------------------------------------------------------

def run_mock(user_query: str) -> str:
    """
    Simulate the agent loop without making real API calls.
    Demonstrates the message structure and tool execution flow.
    """
    header("Raw OpenAI ReAct Agent (MOCK MODE)")
    print(f"Query: {user_query}\n")

    # Simulate iteration 1: search
    step_think("I need to search for information about this topic.", framework="OpenAI")
    step_action("web_search", {"query": user_query}, framework="OpenAI")
    result, _ = execute_tool("web_search", {"query": user_query}, auto_confirm=True)
    step_observe(result, framework="OpenAI")

    # Simulate iteration 2: calculate something
    step_think("Let me compute a relevant value.", framework="OpenAI")
    step_action("calculate", {"expression": "1.1 * 100"}, framework="OpenAI")
    result2, _ = execute_tool("calculate", {"expression": "1.1 * 100"}, auto_confirm=True)
    step_observe(result2, framework="OpenAI")

    # Simulate final answer
    answer = (
        f"Based on my research:\n\n"
        f"Search results provided context about '{user_query}'.\n"
        f"Calculation confirmed: 1.1 * 100 = {result2}\n\n"
        f"[MOCK] In real mode this would be GPT-4's synthesized answer."
    )
    step_final(answer, framework="OpenAI")
    return answer


# ---------------------------------------------------------------------------
# Message structure explainer (educational)
# ---------------------------------------------------------------------------

def explain_message_flow():
    """Print the raw message structure so learners can see what's sent to OpenAI."""
    print("""
RAW OPENAI MESSAGE STRUCTURE
─────────────────────────────────────────────────────────────────

1. SYSTEM message (sets agent persona/instructions):
   {"role": "system", "content": "You are a research assistant..."}

2. USER message (the query):
   {"role": "user", "content": "What is climate change?"}

3. ASSISTANT message with tool_calls (model decides to use a tool):
   {
     "role": "assistant",
     "content": null,
     "tool_calls": [{
       "id": "call_abc123",
       "type": "function",
       "function": {
         "name": "web_search",
         "arguments": '{"query": "climate change"}'
       }
     }]
   }

4. TOOL message (result goes back to the model):
   {
     "role": "tool",
     "tool_call_id": "call_abc123",
     "content": "Search results: ..."
   }

5. ASSISTANT message without tool_calls (final answer):
   {"role": "assistant", "content": "Climate change refers to..."}

The loop is: 3 → 4 → 3 → 4 → ... → 5
─────────────────────────────────────────────────────────────────
""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    parser = argparse.ArgumentParser(description="Raw OpenAI ReAct Agent")
    parser.add_argument("query", nargs="?", default="What are the main causes of climate change? Then save a summary to climate_report.txt", help="Query to run")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no API key needed)")
    parser.add_argument("--explain", action="store_true", help="Show the message flow structure")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    args = parser.parse_args()

    if args.explain:
        explain_message_flow()

    if args.mock or os.getenv("AGENT_MODE") == "mock":
        run_mock(args.query)
    else:
        run_agent(args.query, model=args.model)
