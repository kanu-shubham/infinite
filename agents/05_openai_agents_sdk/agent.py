"""
ReAct Agent — OpenAI Agents SDK
=================================
The OpenAI Agents SDK (released 2025) is OpenAI's official high-level framework.

KEY CONCEPTS UNIQUE TO THIS SDK:
  1. Agent = LLM + instructions + tools (simple dataclass)
  2. Runner.run() handles the loop automatically
  3. Handoffs: agents can TRANSFER control to other agents
  4. Guardrails: input/output validation before/after LLM calls
  5. Tracing: built-in trace logging (OpenAI dashboard or custom)
  6. function_tool decorator: auto-generates schema from type hints

COMPARISON TO OTHERS:
  ┌──────────────────────────────────────────────────────────────┐
  │                     │ Schema source  │ Loop       │ Multi    │
  ├─────────────────────┼────────────────┼────────────┼──────────┤
  │ Raw OpenAI          │ Manual JSON    │ Manual     │ Manual   │
  │ LangGraph           │ Pydantic/hints │ Graph      │ Nodes    │
  │ AutoGen             │ Decorator      │ Chat turns │ Agents   │
  │ OpenAI Agents SDK   │ Type hints     │ Runner     │ Handoffs │
  └──────────────────────────────────────────────────────────────┘

HANDOFFS (killer feature):
  An agent can hand off to a specialized sub-agent:
  - ResearchAgent → ConfirmationAgent (for risky actions)
  - ResearchAgent → WriterAgent (for formatting output)
  The SDK handles transferring context automatically.
"""

import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.tools import execute_tool, TOOLS
from shared.display import header, step_final, step_action, step_observe, step_think

try:
    from agents import Agent, Runner, function_tool, handoff, RunContextWrapper
    from agents.exceptions import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered
    AGENTS_SDK_AVAILABLE = True
except ImportError:
    AGENTS_SDK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Define tools using the @function_tool decorator
# Schema is auto-generated from Python type hints + docstrings
# ---------------------------------------------------------------------------

if AGENTS_SDK_AVAILABLE:

    @function_tool
    def web_search(query: str) -> str:
        """Search the web for information on any topic. Returns titles, URLs, and snippets."""
        step_action("web_search", {"query": query}, framework="Agents SDK")
        result, _ = execute_tool("web_search", {"query": query}, auto_confirm=True)
        step_observe(result, framework="Agents SDK")
        return result

    @function_tool
    def fetch_url(url: str) -> str:
        """Retrieve and read the full content of a URL."""
        step_action("fetch_url", {"url": url}, framework="Agents SDK")
        result, _ = execute_tool("fetch_url", {"url": url}, auto_confirm=True)
        step_observe(result, framework="Agents SDK")
        return result

    @function_tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression. Supports arithmetic, sqrt, sin, cos, log, etc."""
        step_action("calculate", {"expression": expression}, framework="Agents SDK")
        result, _ = execute_tool("calculate", {"expression": expression}, auto_confirm=True)
        step_observe(result, framework="Agents SDK")
        return result

    @function_tool
    def save_file(filename: str, content: str) -> str:
        """[RISKY] Save text content to a file. Requires explicit confirmation before use."""
        step_action("save_file", {"filename": filename, "content": "..."}, framework="Agents SDK")
        result, confirmed = execute_tool("save_file", {"filename": filename, "content": content}, auto_confirm=False)
        step_observe(result, framework="Agents SDK")
        return result

    @function_tool
    def send_email(to: str, subject: str, body: str) -> str:
        """[RISKY] Send an email to a recipient. Requires explicit confirmation before use."""
        step_action("send_email", {"to": to, "subject": subject, "body": body[:50]}, framework="Agents SDK")
        result, confirmed = execute_tool("send_email", {"to": to, "subject": subject, "body": body}, auto_confirm=False)
        step_observe(result, framework="Agents SDK")
        return result


# ---------------------------------------------------------------------------
# Specialized agents for handoffs
# ---------------------------------------------------------------------------

def build_agents():
    """
    Build the agent hierarchy.

    ResearchAgent (main) can hand off to:
      - ConfirmationAgent: explicitly asks the user before risky actions
      - SummaryAgent: formats the final output
    """

    # ── Confirmation Agent (handles risky actions) ────────────────────────
    # When the research agent wants to save a file or send an email,
    # it hands off to this agent which confirms with the user first.
    confirmation_agent = Agent(
        name="ConfirmationAgent",
        instructions="""You handle risky actions that require human approval.

When handed off to you:
1. Clearly describe what action is about to be taken
2. Ask the user for explicit approval
3. If approved, execute the action
4. If denied, report back that the action was cancelled

Always be explicit about what data will be saved or sent.""",
        tools=[save_file, send_email],
    )

    # ── Summary Agent (formats final output) ──────────────────────────────
    summary_agent = Agent(
        name="SummaryAgent",
        instructions="""You format research findings into clear, structured reports.
Structure your response with:
- Executive Summary (2-3 sentences)
- Key Findings (bullet points)
- Sources Used
- Conclusion""",
        tools=[],
    )

    # ── Main Research Agent ────────────────────────────────────────────────
    research_agent = Agent(
        name="ResearchAgent",
        instructions="""You are a research assistant. Use the ReAct pattern.

For information gathering: use web_search and fetch_url.
For calculations: use calculate.
For saving files or sending emails: HAND OFF to ConfirmationAgent.
When ready to present findings: HAND OFF to SummaryAgent.

Always explain your reasoning before calling tools.""",
        tools=[web_search, fetch_url, calculate],
        handoffs=[
            handoff(confirmation_agent, tool_description="Hand off to ConfirmationAgent when the user wants to save a file or send an email. Pass all necessary data."),
            handoff(summary_agent, tool_description="Hand off to SummaryAgent when you have complete research findings and need a formatted report."),
        ],
    )

    return research_agent, confirmation_agent, summary_agent


# ---------------------------------------------------------------------------
# Run the agent
# ---------------------------------------------------------------------------

async def run_agent_async(user_query: str) -> str:
    if not AGENTS_SDK_AVAILABLE:
        raise RuntimeError("openai-agents not installed. Run: pip install openai-agents")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    os.environ["OPENAI_API_KEY"] = api_key

    research_agent, _, _ = build_agents()

    header("OpenAI Agents SDK")

    # Runner.run() handles the entire agent loop including handoffs
    result = await Runner.run(research_agent, input=user_query)

    final = result.final_output
    step_final(str(final), framework="Agents SDK")
    return str(final)


def run_agent(user_query: str) -> str:
    """Synchronous wrapper for run_agent_async."""
    import asyncio
    return asyncio.run(run_agent_async(user_query))


# ---------------------------------------------------------------------------
# Simple single-agent version (no handoffs) for learning
# ---------------------------------------------------------------------------

async def run_simple_async(user_query: str) -> str:
    """
    Simplified version: one agent, all tools, no handoffs.
    Good starting point before learning handoffs.
    """
    if not AGENTS_SDK_AVAILABLE:
        raise RuntimeError("openai-agents not installed.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    os.environ["OPENAI_API_KEY"] = api_key

    agent = Agent(
        name="SimpleResearchAgent",
        instructions="You are a research assistant. Search the web, fetch URLs, and calculate things. For risky tools (save_file, send_email), always confirm with the user first by describing what you'll do.",
        tools=[web_search, fetch_url, calculate, save_file, send_email],
    )

    header("OpenAI Agents SDK (Simple)")
    result = await Runner.run(agent, input=user_query)
    step_final(str(result.final_output), framework="Agents SDK")
    return str(result.final_output)


# ---------------------------------------------------------------------------
# Mock mode
# ---------------------------------------------------------------------------

def run_mock(user_query: str) -> str:
    header("OpenAI Agents SDK (MOCK MODE)")
    print(f"Query: {user_query}\n")
    print("Agent hierarchy:")
    print("  ResearchAgent (main)")
    print("    ├── handoff → ConfirmationAgent (risky actions)")
    print("    └── handoff → SummaryAgent (final report)\n")

    step_action("web_search", {"query": user_query}, framework="Agents SDK")
    result, _ = execute_tool("web_search", {"query": user_query}, auto_confirm=True)
    step_observe(result, framework="Agents SDK")

    print("\n[MOCK] ResearchAgent hands off to SummaryAgent...")
    answer = (
        f"## Research Report\n\n"
        f"**Query**: {user_query}\n\n"
        f"**Summary**: Found relevant information via web search.\n\n"
        f"**Key Findings**:\n- Mock result 1\n- Mock result 2\n\n"
        f"[MOCK] In real mode: Runner.run() would handle the full agent loop + handoffs."
    )
    step_final(answer, framework="Agents SDK")
    return answer


def explain_sdk():
    print("""
OPENAI AGENTS SDK EXPLAINED
─────────────────────────────────────────────────────────────────

Agent = dataclass with:
  name: str
  instructions: str          (system prompt)
  tools: list[function_tool]
  handoffs: list[handoff]    (other agents to transfer to)
  guardrails: list[...]      (input/output validators)

@function_tool decorator:
  - Reads Python type hints → generates JSON schema automatically
  - Reads docstring → uses as tool description
  - No manual schema writing!

  @function_tool
  def web_search(query: str) -> str:
      "Search the web."  # becomes the description
      ...

Runner.run(agent, input="..."):
  - Manages the loop: LLM call → tool execution → LLM call → ...
  - Handles handoffs: when agent calls transfer_to_X(), switches agent
  - Returns RunResult with .final_output and full trace

Handoffs:
  - Agent A can hand off to Agent B mid-conversation
  - Context (message history) is passed along
  - B continues from where A left off

Guardrails (advanced):
  @input_guardrail  → validates user input before LLM sees it
  @output_guardrail → validates LLM output before it's returned

Tracing:
  Every run is traced automatically.
  View in OpenAI dashboard or export to custom collector.
─────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    parser = argparse.ArgumentParser(description="OpenAI Agents SDK")
    parser.add_argument("query", nargs="?", default="Research Python async best practices, then save a cheat-sheet to async_tips.txt")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--simple", action="store_true", help="Use single-agent (no handoffs)")
    args = parser.parse_args()

    if args.explain:
        explain_sdk()

    if args.mock or os.getenv("AGENT_MODE") == "mock":
        run_mock(args.query)
    else:
        import asyncio
        if args.simple:
            asyncio.run(run_simple_async(args.query))
        else:
            asyncio.run(run_agent_async(args.query))
