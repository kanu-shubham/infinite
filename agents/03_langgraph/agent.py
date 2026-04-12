"""
ReAct Agent — LangGraph State Machine
=======================================
LangGraph models the agent as a GRAPH of nodes connected by edges.

WHY THIS IS DIFFERENT from raw API calls:
  - The "loop" becomes an explicit state machine with named nodes
  - State is a TypedDict — inspectable, serializable, resumable
  - You can add checkpointing (pause/resume mid-execution)
  - Conditional edges replace manual if/else routing logic
  - Human-in-the-loop is a first-class concept (interrupt_before)
  - Much easier to debug: you can visualize the graph

GRAPH STRUCTURE:
  ┌─────────┐
  │  START  │
  └────┬────┘
       │
  ┌────▼────────────┐
  │  call_llm       │  ← sends messages to LLM
  └────┬────────────┘
       │
       ├─── tool_calls? ──→ ┌─────────────────┐
       │                    │  check_risky     │  ← is this tool risky?
       │                    └────┬────────────┘
       │                         │
       │              risky? ────┤──→ ┌──────────────────┐
       │                         │    │  human_confirm   │  ← pause for human
       │                         │    └────┬─────────────┘
       │                    safe?│         │
       │                    ─────┘         │
       │                    ↓              ↓
       │                    ┌─────────────────────┐
       │                    │   execute_tools      │  ← run the tool(s)
       │                    └────────┬────────────┘
       │                             │
       └──────── (loop) ─────────────┘
       │
       └─── no tool_calls? ──→ END
"""

import json
import os
import sys
from typing import Annotated, Any, Literal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.tools import execute_tool, TOOLS
from shared.display import step_think, step_action, step_observe, step_final, header, error

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_core.tools import tool as lc_tool
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


SYSTEM_PROMPT = """You are a research assistant. Use the ReAct pattern.
For risky actions (save_file, send_email), explain what you will do before calling."""


# ---------------------------------------------------------------------------
# Agent State — the TypedDict that flows through the graph
# ---------------------------------------------------------------------------

if LANGGRAPH_AVAILABLE:
    from typing import TypedDict

    class AgentState(TypedDict):
        # add_messages is a reducer: new messages are appended, not replaced
        messages: Annotated[list, add_messages]
        # track pending risky tool calls waiting for confirmation
        pending_confirmation: list[dict[str, Any]]
        # user decision for pending confirmations
        confirmed: bool


# ---------------------------------------------------------------------------
# Build LangChain tool wrappers from our shared tool registry
# ---------------------------------------------------------------------------

def build_lc_tools():
    """
    Wrap our plain Python functions as LangChain Tool objects.
    LangGraph uses these for automatic schema generation.
    """
    lc_tools = []
    for name, info in TOOLS.items():
        fn = info["fn"]
        # Create a simple wrapper with the right name/description
        wrapped = lc_tool(fn)
        wrapped.name = name
        wrapped.description = info["description"]
        lc_tools.append(wrapped)
    return lc_tools


# ---------------------------------------------------------------------------
# Graph node functions
# ---------------------------------------------------------------------------

def make_call_llm_node(llm_with_tools):
    """Returns a node function that calls the LLM."""
    def call_llm(state: "AgentState") -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)

        if response.content:
            step_think(str(response.content), framework="LangGraph")

        return {"messages": [response]}
    return call_llm


def check_for_tools(state: "AgentState") -> Literal["execute_tools", "human_confirm", "end"]:
    """
    Conditional edge: inspect the last message to decide next node.

    LangGraph uses the RETURN VALUE of conditional edges to route.
    Strings map to node names.
    """
    last = state["messages"][-1]

    # No tool calls → we're done
    if not getattr(last, "tool_calls", None):
        return "end"

    # Check if any tool call is risky
    for tc in last.tool_calls:
        if TOOLS.get(tc["name"], {}).get("risky", False):
            return "human_confirm"

    return "execute_tools"


def human_confirm_node(state: "AgentState") -> dict:
    """
    Node for human-in-the-loop confirmation of risky actions.

    In production this could send a Slack message, write to a DB,
    or use LangGraph's interrupt() to pause the graph.
    Here we use stdin for simplicity.
    """
    last = state["messages"][-1]
    risky_calls = [tc for tc in last.tool_calls if TOOLS.get(tc["name"], {}).get("risky", False)]

    print("\n" + "=" * 60)
    print("  [LangGraph] HUMAN CONFIRMATION REQUIRED")
    print("=" * 60)
    for tc in risky_calls:
        print(f"  Tool : {tc['name']}")
        print(f"  Args : {json.dumps(tc['args'], indent=4)}")
    print("=" * 60)
    response = input("  Allow these actions? [y/N]: ").strip().lower()
    print("=" * 60 + "\n")

    return {"confirmed": response in ("y", "yes")}


def execute_tools_node(state: "AgentState") -> dict:
    """
    Node that executes all tool calls from the last assistant message.
    Builds ToolMessage results to feed back into the conversation.
    """
    last = state["messages"][-1]
    tool_messages = []
    confirmed = state.get("confirmed", True)  # default True for non-risky tools

    for tc in last.tool_calls:
        tool_name = tc["name"]
        args = tc["args"]
        is_risky = TOOLS.get(tool_name, {}).get("risky", False)

        step_action(tool_name, args, framework="LangGraph")

        if is_risky and not confirmed:
            result = f"Action '{tool_name}' was declined by the user."
        else:
            result, _ = execute_tool(tool_name, args, auto_confirm=True)

        step_observe(result, framework="LangGraph")

        tool_messages.append(
            ToolMessage(content=result, tool_call_id=tc["id"])
        )

    # Reset confirmation state
    return {"messages": tool_messages, "confirmed": True}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph(llm_with_tools):
    """
    Construct and compile the LangGraph state machine.

    compile() validates the graph and returns a runnable.
    You can pass checkpointer=MemorySaver() here for persistence.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("call_llm", make_call_llm_node(llm_with_tools))
    graph.add_node("human_confirm", human_confirm_node)
    graph.add_node("execute_tools", execute_tools_node)

    # Entry point
    graph.add_edge(START, "call_llm")

    # Conditional routing after LLM call
    graph.add_conditional_edges(
        "call_llm",
        check_for_tools,
        {
            "execute_tools": "execute_tools",
            "human_confirm": "human_confirm",
            "end": END,
        },
    )

    # After confirmation → always execute
    graph.add_edge("human_confirm", "execute_tools")

    # After tool execution → back to LLM
    graph.add_edge("execute_tools", "call_llm")

    return graph.compile()


# ---------------------------------------------------------------------------
# Run the agent
# ---------------------------------------------------------------------------

def run_agent(user_query: str, provider: str = "openai", auto_confirm: bool = False) -> str:
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("langgraph/langchain not installed. Run: pip install langgraph langchain-openai langchain-anthropic")

    # Build LLM with tools bound
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        llm = ChatAnthropic(model="claude-haiku-4-5-20251001", api_key=api_key)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

    lc_tools = build_lc_tools()
    llm_with_tools = llm.bind_tools(lc_tools)

    app = build_graph(llm_with_tools)

    header("LangGraph State Machine Agent")

    # Initial state
    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_query)],
        "pending_confirmation": [],
        "confirmed": True,
    }

    # invoke() runs the graph until END node is reached
    final_state = app.invoke(initial_state)

    # Extract the last AI message as the answer
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            step_final(str(msg.content), framework="LangGraph")
            return str(msg.content)

    return "(no final answer)"


# ---------------------------------------------------------------------------
# Mock mode — simulates graph traversal without API calls
# ---------------------------------------------------------------------------

def run_mock(user_query: str) -> str:
    header("LangGraph State Machine Agent (MOCK MODE)")
    print(f"Query: {user_query}\n")
    print("Graph traversal:")
    print("  START → call_llm → [tool_calls found] → execute_tools → call_llm → END\n")

    step_think("I'll search the web for this query.", framework="LangGraph")
    step_action("web_search", {"query": user_query}, framework="LangGraph")
    result, _ = execute_tool("web_search", {"query": user_query}, auto_confirm=True)
    step_observe(result, framework="LangGraph")

    answer = f"[MOCK LangGraph] Research complete for '{user_query}'.\n\nThe graph traversed: START → call_llm → execute_tools → call_llm → END"
    step_final(answer, framework="LangGraph")
    return answer


def explain_graph():
    print("""
LANGGRAPH STATE MACHINE EXPLAINED
─────────────────────────────────────────────────────────────────

State = TypedDict (flows through all nodes):
  {
    "messages": [...],              # conversation history
    "pending_confirmation": [...],  # risky tool calls awaiting approval
    "confirmed": bool               # human's decision
  }

Nodes = Python functions that receive state, return partial updates:
  def call_llm(state) -> {"messages": [new_ai_message]}
  def execute_tools(state) -> {"messages": [tool_result_messages]}
  def human_confirm(state) -> {"confirmed": True/False}

Edges = connections between nodes:
  - Regular edge: always go from A to B
  - Conditional edge: function inspects state → returns node name string

add_messages reducer: instead of replacing messages, it APPENDS.
  This is the magic that makes the conversation history work.

graph.compile() validates:
  - No unreachable nodes
  - All conditional edge values map to valid node names
  - State TypedDict is consistent

For production: pass checkpointer=MemorySaver() to compile() for:
  - Resumable agents (pause mid-execution)
  - Human-in-the-loop via interrupt_before=["human_confirm"]
  - Time-travel debugging (re-run from any checkpoint)
─────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    parser = argparse.ArgumentParser(description="LangGraph ReAct Agent")
    parser.add_argument("query", nargs="?", default="Research React hooks best practices and save key points to react_hooks.txt")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    args = parser.parse_args()

    if args.explain:
        explain_graph()

    if args.mock or os.getenv("AGENT_MODE") == "mock":
        run_mock(args.query)
    else:
        run_agent(args.query, provider=args.provider)
