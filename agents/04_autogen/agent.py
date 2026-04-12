"""
ReAct Agent — AutoGen Multi-Agent Conversation
================================================
AutoGen models agents as CONVERSATIONAL ACTORS that talk to each other.

KEY DIFFERENCE from previous approaches:
  Instead of one agent + tool loop, you have MULTIPLE agents:
  - AssistantAgent: the AI brain (calls tools, reasons)
  - UserProxyAgent: represents the human (executes tools, confirms actions)
  - (Optional) specialized sub-agents for different tasks

HOW IT WORKS:
  1. User sends a message to the UserProxy
  2. UserProxy passes it to the AssistantAgent
  3. AssistantAgent replies with a tool call (Python code block or function call)
  4. UserProxy executes the tool call (it's the "executor")
  5. UserProxy sends the result back to AssistantAgent
  6. Loop until AssistantAgent says "TERMINATE"

PARADIGM SHIFT:
  ┌─────────────────────────────────────────────────────────┐
  │  Raw API / LangGraph   │  AutoGen                       │
  ├────────────────────────┼────────────────────────────────┤
  │  One agent + loop      │  Multiple agents conversing    │
  │  You manage the loop   │  Framework manages turns       │
  │  Tools = schemas       │  Tools = registered functions  │
  │  Human = special node  │  Human = UserProxyAgent        │
  └────────────────────────┴────────────────────────────────┘

HUMAN-IN-THE-LOOP in AutoGen:
  UserProxyAgent has human_input_mode:
  - "NEVER": fully autonomous
  - "TERMINATE": ask at the end
  - "ALWAYS": ask before every action  ← best for risky actions
"""

import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.tools import execute_tool, TOOLS, web_search, fetch_url, calculate, save_file, send_email
from shared.display import header, step_final, step_think, step_action, step_observe

try:
    import autogen
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False


SYSTEM_MESSAGE = """You are a research assistant. Use the registered tools to:
1. Search for information (web_search)
2. Read URLs (fetch_url)
3. Do calculations (calculate)
4. Save results or send emails ONLY after getting user approval (save_file, send_email)

When you have a complete answer, end your message with exactly: TERMINATE"""


# ---------------------------------------------------------------------------
# Register our shared tools with AutoGen
# ---------------------------------------------------------------------------

def register_tools_with_agents(assistant, user_proxy):
    """
    AutoGen uses a decorator pattern to register tools.
    The executor (user_proxy) runs tools; the caller (assistant) decides when.
    """

    # Safe tools — no confirmation needed
    @user_proxy.register_for_execution()
    @assistant.register_for_llm(description=TOOLS["web_search"]["description"])
    def ag_web_search(query: str) -> str:
        result, _ = execute_tool("web_search", {"query": query}, auto_confirm=True)
        step_action("web_search", {"query": query}, framework="AutoGen")
        step_observe(result, framework="AutoGen")
        return result

    @user_proxy.register_for_execution()
    @assistant.register_for_llm(description=TOOLS["fetch_url"]["description"])
    def ag_fetch_url(url: str) -> str:
        result, _ = execute_tool("fetch_url", {"url": url}, auto_confirm=True)
        step_action("fetch_url", {"url": url}, framework="AutoGen")
        step_observe(result, framework="AutoGen")
        return result

    @user_proxy.register_for_execution()
    @assistant.register_for_llm(description=TOOLS["calculate"]["description"])
    def ag_calculate(expression: str) -> str:
        result, _ = execute_tool("calculate", {"expression": expression}, auto_confirm=True)
        step_action("calculate", {"expression": expression}, framework="AutoGen")
        step_observe(result, framework="AutoGen")
        return result

    # Risky tools — confirmation is handled by UserProxyAgent with human_input_mode="ALWAYS"
    # The UserProxy will pause and ask before executing these
    @user_proxy.register_for_execution()
    @assistant.register_for_llm(description=TOOLS["save_file"]["description"])
    def ag_save_file(filename: str, content: str) -> str:
        # With human_input_mode="ALWAYS" on user_proxy, the user has already
        # confirmed this execution. We can proceed.
        result, _ = execute_tool("save_file", {"filename": filename, "content": content}, auto_confirm=True)
        step_action("save_file", {"filename": filename, "content": "..."}, framework="AutoGen")
        step_observe(result, framework="AutoGen")
        return result

    @user_proxy.register_for_execution()
    @assistant.register_for_llm(description=TOOLS["send_email"]["description"])
    def ag_send_email(to: str, subject: str, body: str) -> str:
        result, _ = execute_tool("send_email", {"to": to, "subject": subject, "body": body}, auto_confirm=True)
        step_action("send_email", {"to": to, "subject": subject, "body": body[:50]}, framework="AutoGen")
        step_observe(result, framework="AutoGen")
        return result


# ---------------------------------------------------------------------------
# Run the multi-agent conversation
# ---------------------------------------------------------------------------

def run_agent(user_query: str, model: str = "gpt-4o-mini", human_input_mode: str = "TERMINATE") -> str:
    """
    human_input_mode controls when the UserProxy asks for human input:
      "NEVER"    — fully autonomous, no prompts
      "TERMINATE" — only ask when the conversation ends (review before TERMINATE)
      "ALWAYS"   — ask before executing every tool call (use for risky agents)
    """
    if not AUTOGEN_AVAILABLE:
        raise RuntimeError("pyautogen not installed. Run: pip install pyautogen")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    config_list = [{"model": model, "api_key": api_key}]
    llm_config = {"config_list": config_list, "cache_seed": None}

    # ── Create the two agents ──────────────────────────────────────────────
    assistant = autogen.AssistantAgent(
        name="ResearchAssistant",
        system_message=SYSTEM_MESSAGE,
        llm_config=llm_config,
    )

    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        human_input_mode=human_input_mode,
        # is_termination_msg: stop the conversation when we see TERMINATE
        is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
        code_execution_config=False,  # we use function tools, not code execution
        max_consecutive_auto_reply=10,
    )

    # Register tools with both agents
    register_tools_with_agents(assistant, user_proxy)

    header("AutoGen Multi-Agent Conversation")
    print(f"  Assistant: ResearchAssistant")
    print(f"  Executor : UserProxy (human_input_mode={human_input_mode})\n")

    # ── Start the conversation ─────────────────────────────────────────────
    # This kicks off the multi-turn conversation loop
    user_proxy.initiate_chat(
        assistant,
        message=user_query,
        max_turns=15,
    )

    # Extract final message from the assistant
    chat_history = user_proxy.chat_messages.get(assistant, [])
    for msg in reversed(chat_history):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            # Remove TERMINATE marker
            final = content.replace("TERMINATE", "").strip()
            if final:
                step_final(final, framework="AutoGen")
                return final

    return "(conversation ended without final answer)"


# ---------------------------------------------------------------------------
# Multi-agent pattern: Research team with specialized agents
# ---------------------------------------------------------------------------

def run_research_team(user_query: str) -> str:
    """
    Advanced AutoGen pattern: a team of specialized agents.

    ResearchAgent → searches and retrieves
    AnalystAgent  → analyzes and synthesizes
    WriterAgent   → formats the final output
    UserProxy     → coordinates and executes
    """
    if not AUTOGEN_AVAILABLE:
        raise RuntimeError("pyautogen not installed.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    config_list = [{"model": "gpt-4o-mini", "api_key": api_key}]
    llm_config = {"config_list": config_list}

    researcher = autogen.AssistantAgent(
        name="Researcher",
        system_message="You are a web researcher. Use web_search and fetch_url to gather information. Pass findings to the Analyst.",
        llm_config=llm_config,
    )

    analyst = autogen.AssistantAgent(
        name="Analyst",
        system_message="You are a data analyst. Analyze the research findings, identify key insights, and pass to the Writer.",
        llm_config=llm_config,
    )

    writer = autogen.AssistantAgent(
        name="Writer",
        system_message="You are a technical writer. Format the analysis into a clear report. End with TERMINATE.",
        llm_config=llm_config,
    )

    user_proxy = autogen.UserProxyAgent(
        name="UserProxy",
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
        code_execution_config=False,
    )

    # Group chat: all agents talk in a round-robin
    group_chat = autogen.GroupChat(
        agents=[user_proxy, researcher, analyst, writer],
        messages=[],
        max_round=12,
        speaker_selection_method="round_robin",
    )
    manager = autogen.GroupChatManager(groupchat=group_chat, llm_config=llm_config)

    header("AutoGen Research Team (GroupChat)")
    user_proxy.initiate_chat(manager, message=user_query)
    return "Research team conversation complete."


# ---------------------------------------------------------------------------
# Mock mode
# ---------------------------------------------------------------------------

def run_mock(user_query: str) -> str:
    header("AutoGen Multi-Agent Conversation (MOCK MODE)")
    print(f"Query: {user_query}\n")
    print("Agents: ResearchAssistant ↔ UserProxy\n")
    print("Conversation flow:")
    print("  UserProxy → ResearchAssistant: 'Research climate change'")
    print("  ResearchAssistant → UserProxy: 'I'll call web_search()'")
    print("  UserProxy executes web_search() and returns result")
    print("  ResearchAssistant → UserProxy: 'TERMINATE — here is the answer'\n")

    step_action("web_search", {"query": user_query}, framework="AutoGen")
    result, _ = execute_tool("web_search", {"query": user_query}, auto_confirm=True)
    step_observe(result, framework="AutoGen")

    answer = f"[MOCK AutoGen] Multi-agent research complete for '{user_query}'.\n\nIn real mode: ResearchAssistant and UserProxy would have a full conversation."
    step_final(answer, framework="AutoGen")
    return answer


def explain_autogen():
    print("""
AUTOGEN MULTI-AGENT EXPLAINED
─────────────────────────────────────────────────────────────────

Core concept: Agents TALK to each other. Each has a role.

AssistantAgent:
  - Has LLM (GPT-4, Claude, etc.)
  - Decides WHAT to do (calls tools, reasons)
  - Registers: @assistant.register_for_llm(description="...")

UserProxyAgent:
  - Represents the human / execution environment
  - Executes tool calls
  - Registers: @user_proxy.register_for_execution()
  - Controls human input: "NEVER" | "TERMINATE" | "ALWAYS"

human_input_mode options:
  "NEVER"     → fully autonomous (good for safe workflows)
  "TERMINATE" → ask human before conversation ends (review final answer)
  "ALWAYS"    → ask before EVERY action (best for risky agents)

GroupChat (advanced):
  Multiple agents take turns in a round-robin (or LLM-selected order).
  A GroupChatManager orchestrates the conversation.
  Example team: Researcher → Analyst → Writer → UserProxy

Key advantage: Each agent can have different:
  - LLM model (GPT-4 for reasoning, GPT-3.5 for drafting)
  - System prompt / persona
  - Tool access
  - Human input mode
─────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    parser = argparse.ArgumentParser(description="AutoGen Multi-Agent")
    parser.add_argument("query", nargs="?", default="Research climate change impacts and save a summary report to climate_summary.txt")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--team", action="store_true", help="Use GroupChat research team")
    parser.add_argument("--human-input", default="TERMINATE", choices=["NEVER", "TERMINATE", "ALWAYS"])
    args = parser.parse_args()

    if args.explain:
        explain_autogen()

    if args.mock or os.getenv("AGENT_MODE") == "mock":
        run_mock(args.query)
    elif args.team:
        run_research_team(args.query)
    else:
        run_agent(args.query, human_input_mode=args.human_input)
