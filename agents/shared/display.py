"""
Pretty-print utilities for agent traces.
Shows the ReAct loop steps: Thought → Action → Observation.
"""

import json
from typing import Any

# Use rich if available, otherwise plain text
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    RICH = True
    console = Console()
except ImportError:
    RICH = False


def step_think(text: str, framework: str = ""):
    label = f"[{framework}] " if framework else ""
    if RICH:
        console.print(Panel(text, title=f"{label}Thought", border_style="cyan", expand=False))
    else:
        print(f"\n{'─'*60}")
        print(f"  {label}THOUGHT")
        print(f"{'─'*60}")
        print(f"  {text}")


def step_action(tool_name: str, args: dict[str, Any], framework: str = ""):
    label = f"[{framework}] " if framework else ""
    args_str = json.dumps(args, indent=2)
    if RICH:
        console.print(Panel(f"Tool: [bold yellow]{tool_name}[/]\nArgs: {args_str}", title=f"{label}Action", border_style="yellow", expand=False))
    else:
        print(f"\n{'─'*60}")
        print(f"  {label}ACTION  →  {tool_name}")
        print(f"{'─'*60}")
        print(f"  Args: {args_str}")


def step_observe(result: str, framework: str = ""):
    label = f"[{framework}] " if framework else ""
    # truncate very long results
    display = result if len(result) < 500 else result[:497] + "..."
    if RICH:
        console.print(Panel(display, title=f"{label}Observation", border_style="green", expand=False))
    else:
        print(f"\n{'─'*60}")
        print(f"  {label}OBSERVATION")
        print(f"{'─'*60}")
        print(f"  {display}")


def step_final(answer: str, framework: str = ""):
    label = f"[{framework}] " if framework else ""
    if RICH:
        console.print(Panel(answer, title=f"{label}Final Answer", border_style="bold magenta", expand=False))
    else:
        print(f"\n{'='*60}")
        print(f"  {label}FINAL ANSWER")
        print(f"{'='*60}")
        print(f"  {answer}")
        print(f"{'='*60}\n")


def header(title: str):
    if RICH:
        console.rule(f"[bold blue]{title}[/]")
    else:
        print(f"\n{'#'*60}")
        print(f"  {title}")
        print(f"{'#'*60}\n")


def error(msg: str, framework: str = ""):
    label = f"[{framework}] " if framework else ""
    if RICH:
        console.print(Panel(msg, title=f"{label}ERROR", border_style="bold red", expand=False))
    else:
        print(f"\n!!! {label}ERROR: {msg} !!!\n")
