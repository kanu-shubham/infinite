"""
Framework Comparison Runner
============================
Runs the same query through all four agent frameworks and prints a
side-by-side comparison of reliability, debuggability, and boilerplate.

Usage:
  python run_comparison.py --mock              # no API keys needed
  python run_comparison.py --mock --query "..."
  python run_comparison.py                     # needs API keys in .env

What's compared:
  1. Boilerplate: lines of code to build the agent
  2. Debuggability: how visible the loop internals are
  3. Error recovery: what happens on tool failure
  4. Human-in-the-loop: how risky actions are handled
  5. Multi-agent: how easily you add more agents
"""

import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

sys.path.insert(0, os.path.dirname(__file__))

from shared.display import header

# Try importing each framework's agent
def _try_import(module_path: str) -> tuple:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("agent", module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, None
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Comparison data collection
# ---------------------------------------------------------------------------

@dataclass
class FrameworkResult:
    name: str
    answer: str = ""
    duration_sec: float = 0.0
    error: str = ""
    success: bool = False


def run_with_timing(name: str, fn: Callable, query: str) -> FrameworkResult:
    result = FrameworkResult(name=name)
    start = time.time()
    try:
        result.answer = fn(query)
        result.success = True
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        result.success = False
    result.duration_sec = time.time() - start
    return result


# ---------------------------------------------------------------------------
# Framework scorecard (static analysis — not runtime)
# ---------------------------------------------------------------------------

FRAMEWORK_ANALYSIS = {
    "Raw OpenAI": {
        "boilerplate": "Medium (50-80 lines for the loop)",
        "debuggability": "High — you see every message dict",
        "error_recovery": "Manual — catch exceptions, inject error as tool result",
        "human_in_loop": "Manual — check tool name, call input() yourself",
        "multi_agent": "Manual — you manage multiple client instances",
        "learning_value": "Essential — teaches the fundamentals",
        "production_ready": "Yes, with effort",
        "best_for": "Learning, full control, custom protocols",
    },
    "Raw Anthropic": {
        "boilerplate": "Medium (50-80 lines, similar to OpenAI)",
        "debuggability": "High — typed response blocks are clear",
        "error_recovery": "Manual — same approach as OpenAI",
        "human_in_loop": "Manual — same approach",
        "multi_agent": "Manual",
        "learning_value": "Essential — teaches Anthropic's block-based format",
        "production_ready": "Yes, with effort",
        "best_for": "Anthropic-specific features (extended thinking, vision)",
    },
    "LangGraph": {
        "boilerplate": "High (100+ lines) but very structured",
        "debuggability": "Excellent — visualize graph, inspect state at any node",
        "error_recovery": "Built-in retry edges, error nodes",
        "human_in_loop": "First-class: interrupt_before, checkpointing",
        "multi_agent": "Natural: multiple graphs, supervisor patterns",
        "learning_value": "High — teaches state machine thinking",
        "production_ready": "Excellent — used in production widely",
        "best_for": "Complex workflows, long-running agents, resumable tasks",
    },
    "AutoGen": {
        "boilerplate": "Low-Medium (register tools with decorators)",
        "debuggability": "Good — chat transcript is easy to follow",
        "error_recovery": "Via conversation: assistant can ask for retry",
        "human_in_loop": "Built-in: human_input_mode parameter",
        "multi_agent": "Excellent — designed for this (GroupChat)",
        "learning_value": "High — teaches multi-agent conversation design",
        "production_ready": "Good — Microsoft-backed",
        "best_for": "Team-of-agents, code execution workflows",
    },
    "OpenAI Agents SDK": {
        "boilerplate": "Low (@function_tool auto-generates schemas)",
        "debuggability": "Good — built-in tracing to OpenAI dashboard",
        "error_recovery": "Built-in retry, guardrails catch bad outputs",
        "human_in_loop": "Via handoff to ConfirmationAgent",
        "multi_agent": "Excellent — handoffs are first-class",
        "learning_value": "Medium — abstracts too much to learn from",
        "production_ready": "Excellent — OpenAI-native",
        "best_for": "OpenAI ecosystem, quick production agents",
    },
}


def print_scorecard():
    """Print a formatted comparison table."""
    print("\n")
    header("FRAMEWORK COMPARISON SCORECARD")

    dimensions = [
        "boilerplate", "debuggability", "error_recovery",
        "human_in_loop", "multi_agent", "best_for",
    ]

    col_width = 30
    name_width = 20

    # Header row
    print(f"\n{'Dimension':<22}", end="")
    for fw in FRAMEWORK_ANALYSIS:
        short = fw.replace("Raw ", "").replace(" SDK", "")
        print(f"{short:<{col_width}}", end="")
    print()
    print("─" * (22 + col_width * len(FRAMEWORK_ANALYSIS)))

    for dim in dimensions:
        label = dim.replace("_", " ").title()
        print(f"\n{label:<22}", end="")
        for fw, scores in FRAMEWORK_ANALYSIS.items():
            val = scores.get(dim, "—")
            # Truncate for display
            if len(val) > col_width - 2:
                val = val[:col_width - 5] + "..."
            print(f"{val:<{col_width}}", end="")
        print()

    print("\n" + "─" * (22 + col_width * len(FRAMEWORK_ANALYSIS)))


def print_when_to_use():
    """Print decision guide."""
    print("""
WHEN TO USE EACH FRAMEWORK
─────────────────────────────────────────────────────────────────

Use RAW API (OpenAI or Anthropic) when:
  ✓ You're learning how agents work (START HERE)
  ✓ You need total control over the message format
  ✓ You're building a non-standard agent protocol
  ✓ Minimizing dependencies is critical

Use LANGGRAPH when:
  ✓ Your workflow has multiple distinct phases/states
  ✓ You need agents that can be paused and resumed
  ✓ You need strong observability and debugging
  ✓ Long-running workflows (minutes to hours)
  ✓ You need time-travel debugging

Use AUTOGEN when:
  ✓ You want multiple AI agents collaborating
  ✓ Code execution is part of the workflow
  ✓ You like the "chat between agents" mental model
  ✓ Research/analysis tasks with distinct roles

Use OPENAI AGENTS SDK when:
  ✓ You're OpenAI-native and want the simplest setup
  ✓ You need handoffs between specialized agents
  ✓ You want built-in tracing on OpenAI's platform
  ✓ Production deployment with minimal overhead

─────────────────────────────────────────────────────────────────
""")


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------

def run_comparison(query: str, mock: bool = True):
    base = os.path.dirname(__file__)

    runners = [
        ("Raw OpenAI",       os.path.join(base, "01_raw_openai", "agent.py"),       "run_mock" if mock else "run_agent"),
        ("Raw Anthropic",    os.path.join(base, "02_raw_anthropic", "agent.py"),    "run_mock" if mock else "run_agent"),
        ("LangGraph",        os.path.join(base, "03_langgraph", "agent.py"),        "run_mock" if mock else "run_agent"),
        ("AutoGen",          os.path.join(base, "04_autogen", "agent.py"),          "run_mock" if mock else "run_agent"),
        ("OpenAI Agents SDK",os.path.join(base, "05_openai_agents_sdk", "agent.py"),"run_mock" if mock else "run_agent"),
    ]

    results = []

    for fw_name, agent_path, fn_name in runners:
        print(f"\n{'='*60}")
        print(f"  Running: {fw_name}")
        print(f"{'='*60}")

        mod, import_err = _try_import(agent_path)
        if import_err:
            r = FrameworkResult(name=fw_name, error=f"Import failed: {import_err}", success=False)
            results.append(r)
            print(f"  SKIPPED: {import_err}")
            continue

        fn = getattr(mod, fn_name, None)
        if fn is None:
            r = FrameworkResult(name=fw_name, error=f"Function '{fn_name}' not found", success=False)
            results.append(r)
            continue

        r = run_with_timing(fw_name, fn, query)
        results.append(r)

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n\n")
    header("RESULTS SUMMARY")
    print(f"\n{'Framework':<25} {'Status':<10} {'Time':>8}  Answer preview")
    print("─" * 80)
    for r in results:
        status = "OK" if r.success else "FAILED"
        preview = r.answer[:40].replace("\n", " ") if r.success else r.error[:40]
        print(f"{r.name:<25} {status:<10} {r.duration_sec:>6.2f}s  {preview}")

    # ── Scorecard ─────────────────────────────────────────────────────────
    print_scorecard()
    print_when_to_use()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run all four agents and compare")
    parser.add_argument("--query", default="What are the main causes of climate change?", help="Research query to run")
    parser.add_argument("--mock", action="store_true", default=True, help="Mock mode (no API keys needed)")
    parser.add_argument("--real", action="store_true", help="Real mode (requires API keys)")
    parser.add_argument("--scorecard", action="store_true", help="Just print the scorecard, don't run agents")
    args = parser.parse_args()

    if args.scorecard:
        print_scorecard()
        print_when_to_use()
    else:
        use_mock = not args.real
        run_comparison(query=args.query, mock=use_mock)
