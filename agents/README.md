# Multi-Tool ReAct Agent — 4 Framework Comparison

A research agent built **4 times** using different frameworks.
Same agent, same tools, different paradigms — learn by comparing.

## What the Agent Does

```
User Query
    ↓
Think → Search web → Observe results
    ↓
Think → Fetch URL → Observe content
    ↓
Think → Calculate → Observe result
    ↓
Think → [RISKY] Save file → ⚠️ Ask human → Execute (or cancel)
    ↓
Final Answer
```

## Project Structure

```
agents/
├── shared/
│   ├── tools.py          ← Tool implementations (used by all frameworks)
│   └── display.py        ← Pretty-print utilities
│
├── 01_raw_openai/
│   └── agent.py          ← Raw OpenAI function calling, manual loop
│
├── 02_raw_anthropic/
│   └── agent.py          ← Raw Anthropic tool use, manual loop
│
├── 03_langgraph/
│   └── agent.py          ← LangGraph state machine (StateGraph)
│
├── 04_autogen/
│   └── agent.py          ← AutoGen multi-agent conversation
│
├── 05_openai_agents_sdk/
│   └── agent.py          ← OpenAI Agents SDK with handoffs
│
├── run_comparison.py     ← Run all 4, compare results
├── requirements.txt
└── .env.example
```

## Quick Start (Mock Mode — No API Keys)

```bash
cd agents

# Install dependencies
pip install -r requirements.txt

# Run all 4 frameworks in mock mode
python run_comparison.py --mock

# Just see the comparison scorecard
python run_comparison.py --scorecard

# Run individual frameworks
python 01_raw_openai/agent.py --mock
python 02_raw_anthropic/agent.py --mock
python 03_langgraph/agent.py --mock
python 04_autogen/agent.py --mock
python 05_openai_agents_sdk/agent.py --mock
```

## Real Mode (With API Keys)

```bash
cp .env.example .env
# Edit .env and add your keys

# Run a specific agent
python 01_raw_openai/agent.py "Research climate change and save a report to climate.txt"
python 02_raw_anthropic/agent.py "Research Python async and save a guide to async.txt"
python 03_langgraph/agent.py "Research React hooks best practices"
python 04_autogen/agent.py "Research climate change" --human-input ALWAYS
python 05_openai_agents_sdk/agent.py "Research Python async"

# Run all and compare
python run_comparison.py --real
```

## Learn the Fundamentals

Each agent has an `--explain` flag that prints the underlying concepts:

```bash
python 01_raw_openai/agent.py --explain     # OpenAI message format
python 02_raw_anthropic/agent.py --explain  # Anthropic block format  
python 03_langgraph/agent.py --explain      # State machine concepts
python 04_autogen/agent.py --explain        # Multi-agent conversation
python 05_openai_agents_sdk/agent.py --explain  # SDK concepts
```

## Tools Available

| Tool | Risky? | Description |
|------|--------|-------------|
| `web_search(query)` | No | Search the web |
| `fetch_url(url)` | No | Retrieve URL content |
| `calculate(expr)` | No | Evaluate math expressions |
| `save_file(filename, content)` | **Yes** | Write to disk — asks for confirmation |
| `send_email(to, subject, body)` | **Yes** | Send email — asks for confirmation |

Risky tools trigger human confirmation before execution.

## Framework Comparison

| | Raw OpenAI | Raw Anthropic | LangGraph | AutoGen | Agents SDK |
|---|---|---|---|---|---|
| **Boilerplate** | Medium | Medium | High | Low-Med | Low |
| **Debuggability** | High | High | Excellent | Good | Good |
| **Error recovery** | Manual | Manual | Built-in | Via chat | Built-in |
| **Human-in-loop** | Manual | Manual | First-class | Built-in | Via handoff |
| **Multi-agent** | Manual | Manual | Natural | Excellent | Excellent |
| **Learning value** | Essential | Essential | High | High | Medium |

## The ReAct Loop

```
┌─────────────────────────────────────────────────┐
│  THINK: What do I need to find out?             │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│  ACT: Call a tool (web_search, fetch_url, ...)  │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────▼──────────────┐
          │  RISKY?                   │
          │  YES → Ask human          │
          │  NO  → Execute            │
          └────────────┬──────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│  OBSERVE: What did the tool return?             │
└──────────────────────┬──────────────────────────┘
                       │
              ┌────────▼──────────┐
              │  Done?            │
              │  YES → Answer     │
              │  NO  → THINK      │
              └───────────────────┘
```
