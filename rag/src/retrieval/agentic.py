"""
Agentic RAG — LLM-directed retrieval with tool use
(sometimes called "RAG Agent" or "Agentic Search")

What is Agentic RAG?
--------------------
Standard RAG is a fixed pipeline: embed → retrieve → generate.
The LLM has no say in HOW retrieval happens.

Agentic RAG gives the LLM control:
  - It decides WHEN to retrieve (maybe the question is answerable from
    context and doesn't need any retrieval)
  - It decides HOW MANY rounds to retrieve (one hop or five)
  - It decides WHAT to search for (it writes its own sub-queries)
  - It decides WHICH tool to use (semantic search vs keyword vs date filter)

This is essentially ReAct (Reason + Act) applied to RAG.

Tool loop
---------
  Thought: I need to find information about X.
  Action: search_documents(query="X")
  Observation: [retrieved chunks]
  Thought: I found X but I still need Y.
  Action: search_documents(query="Y")
  Observation: [more chunks]
  Thought: I have enough information now.
  Final Answer: ...

Tools provided here
-------------------
  search_documents(query, k=5)
    — semantic vector search, the workhorse

  search_by_keyword(keyword, k=5)
    — exact substring match, useful for IDs, names, codes

  search_by_source(source, k=5)
    — filter to chunks from a specific doc/URL, useful for
      "What does the privacy policy say about..."

  answer_directly(answer)
    — agent signals it's done; ends the tool loop

Agentic RAG vs Multi-hop
------------------------
Multi-hop retrieval is hard-coded: always N hops, always passes the
retrieved context as the next query.  It has no ability to decide
"I need a keyword search here" or "I already have enough context."

Agentic RAG is flexible: the agent writes its own queries, picks its
own tools, and decides when to stop.  More powerful but also less
predictable and harder to debug.

When to use Agentic RAG
-----------------------
- Complex multi-step questions ("Compare the revenue growth of
  company A and company B over the last 3 years")
- When you have heterogeneous retrieval strategies (semantic +
  keyword + date-filtered)
- When you want the system to handle both simple and complex
  questions gracefully (it won't waste hops on simple lookups)
- In conversational/chat interfaces where context evolves
"""

from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, List, Optional

import anthropic

from src.config import config
from src.documents import Chunk, RetrievalResult
from src.generation.generator import Generator, RAGResponse
from src.embeddings.base import BaseEmbedder
from src.vectorstore.memory import InMemoryVectorStore
from src.retrieval.base import BaseRAG


# ---------------------------------------------------------------------------
# Tool definitions (Claude tool_use format)
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "search_documents",
        "description": (
            "Search the document corpus using semantic (vector) similarity. "
            "Use this for conceptual or topic-based queries where you want "
            "passages that are semantically related to the question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to embed and retrieve for.",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_by_keyword",
        "description": (
            "Search for chunks containing an exact keyword or phrase. "
            "Use this for product names, IDs, technical terms, or when "
            "semantic search would dilute the meaning of a specific term."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Exact keyword or phrase to search for (case-insensitive).",
                },
                "k": {
                    "type": "integer",
                    "description": "Max results to return.",
                    "default": 5,
                },
            },
            "required": ["keyword"],
        },
    },
    {
        "name": "search_by_source",
        "description": (
            "Retrieve chunks from a specific document source or URL. "
            "Use when the user asks about a specific document: "
            "'What does the 2024 annual report say about...'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Document source identifier or URL substring to match.",
                },
                "k": {
                    "type": "integer",
                    "description": "Max results to return.",
                    "default": 5,
                },
            },
            "required": ["source"],
        },
    },
    {
        "name": "answer_directly",
        "description": (
            "Provide the final answer to the user's question. "
            "Call this when you have gathered sufficient context "
            "and are ready to synthesise a complete answer. "
            "Do NOT call other tools after this."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The complete, well-cited answer to the user's question.",
                },
            },
            "required": ["answer"],
        },
    },
]


_AGENT_SYSTEM = textwrap.dedent("""\
    You are a research assistant with access to a document corpus.

    To answer the user's question:
    1. Use the available search tools to find relevant information.
    2. Think step-by-step about what information you need.
    3. Make multiple search calls if necessary to gather complete context.
    4. When you have enough information, call answer_directly() with a
       comprehensive, well-supported answer that cites the sources found.

    Guidelines:
    - Search iteratively — refine your queries based on what you find.
    - Use search_by_keyword for specific names, IDs, or technical terms.
    - Use search_by_source when you need info from a particular document.
    - Use search_documents for conceptual or thematic questions.
    - Do not repeat the same search query twice.
    - If after 3+ searches you still can't find the answer, call
      answer_directly() and be transparent about the uncertainty.
    - Always call answer_directly() as the last action.
""")


class AgenticRAG(BaseRAG):
    """
    Tool-using RAG agent that decides its own retrieval strategy.

    The agent runs a ReAct loop (max_iterations steps) using Claude's
    tool_use feature.  It can call search tools multiple times with
    different queries before synthesising a final answer.

    Parameters
    ----------
    max_iterations : maximum number of tool calls before forcing an answer
    """

    def __init__(self, *args, max_iterations: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self._max_iterations = max_iterations

    def run(self, query: str) -> RAGResponse:
        messages = [{"role": "user", "content": query}]
        all_results: List[RetrievalResult] = []
        final_answer: Optional[str] = None

        for iteration in range(self._max_iterations):
            resp = self._client.messages.create(
                model=config.model,
                max_tokens=1024,
                system=_AGENT_SYSTEM,
                tools=_TOOLS,
                messages=messages,
            )

            # Collect tool calls from this response turn
            tool_calls = [b for b in resp.content if b.type == "tool_use"]

            if not tool_calls:
                # No tool calls — model responded directly (shouldn't happen
                # given our system prompt, but handle it gracefully)
                text_blocks = [b for b in resp.content if hasattr(b, "text")]
                if text_blocks:
                    final_answer = text_blocks[0].text
                break

            # Append assistant's turn
            messages.append({"role": "assistant", "content": resp.content})

            # Execute each tool call and collect results
            tool_results = []
            for tool_call in tool_calls:
                tool_input: Dict[str, Any] = tool_call.input
                tool_output, is_done = self._execute_tool(
                    tool_call.name, tool_input, all_results
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": tool_output,
                })
                if is_done:
                    final_answer = tool_input.get("answer", "")
                    break

            messages.append({"role": "user", "content": tool_results})

            if final_answer is not None:
                break

        # If the agent never called answer_directly, generate from gathered context
        if final_answer is None:
            rag_resp = self.generator.generate(
                query=query,
                results=all_results[:config.top_k],
                pattern="agentic",
            )
            rag_resp.metadata["iterations"] = iteration + 1
            rag_resp.metadata["agent_answered"] = False
            return rag_resp

        # Build a RAGResponse from the agent's answer and accumulated chunks
        rag_resp = RAGResponse(
            query=query,
            answer=final_answer,
            citations=[],  # agent's answer may reference sources inline
            retrieval_results=all_results,
            pattern="agentic",
            metadata={
                "iterations": iteration + 1,
                "agent_answered": True,
                "n_chunks_retrieved": len(all_results),
            },
        )
        return rag_resp

    def _execute_tool(
        self,
        name: str,
        tool_input: Dict[str, Any],
        all_results: List[RetrievalResult],
    ):
        """
        Execute a tool call and return (output_string, is_terminal).
        is_terminal=True when answer_directly is called.
        """
        if name == "search_documents":
            query = tool_input["query"]
            k = int(tool_input.get("k", 5))
            results = self._retrieve(query, k=k)
            all_results.extend(results)
            return self._format_results(results), False

        elif name == "search_by_keyword":
            keyword = tool_input["keyword"].lower()
            k = int(tool_input.get("k", 5))
            all_chunks = self.store.all_chunks()
            matches = [
                c for c in all_chunks
                if keyword in c.content.lower()
            ][:k]
            results = [RetrievalResult(chunk=c, score=1.0) for c in matches]
            all_results.extend(results)
            return self._format_results(results), False

        elif name == "search_by_source":
            source = tool_input["source"].lower()
            k = int(tool_input.get("k", 5))
            all_chunks = self.store.all_chunks()
            matches = [
                c for c in all_chunks
                if source in (c.doc_source or "").lower()
                or source in (c.doc_title or "").lower()
            ][:k]
            results = [RetrievalResult(chunk=c, score=1.0) for c in matches]
            all_results.extend(results)
            return self._format_results(results), False

        elif name == "answer_directly":
            return "", True  # is_terminal

        return f"Unknown tool: {name}", False

    def _format_results(self, results: List[RetrievalResult]) -> str:
        if not results:
            return "No relevant results found."
        parts = []
        for i, r in enumerate(results):
            src = r.chunk.doc_title or r.chunk.doc_source or "unknown"
            parts.append(f"[{i+1}] Source: {src}\n{r.chunk.content[:400]}")
        return "\n\n".join(parts)
