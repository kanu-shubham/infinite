#!/usr/bin/env python3
"""
RAG System — Interactive CLI

Usage
-----
  # Set your API key first
  export ANTHROPIC_API_KEY=sk-ant-...

  # Run interactive CLI
  python main.py

  # Query with a specific pattern
  python main.py --pattern hyde --query "What is BERT?"

  # Run all patterns on a query and compare
  python main.py --compare --query "How does attention work?"

  # Include metric evaluation
  python main.py --evaluate --query "What are the key RAG patterns?"

  # Index custom documents
  python main.py --docs-dir ./my_docs/ --query "..."
"""

import argparse
import os
import sys
from pathlib import Path

# Allow running from the rag/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import config
from src.documents import Document
from src.pipeline import RAGPipeline, PATTERNS


def load_documents_from_dir(docs_dir: str) -> list[Document]:
    """Load .txt files from a directory as Documents."""
    docs = []
    docs_path = Path(docs_dir)
    for txt_file in sorted(docs_path.glob("*.txt")):
        content = txt_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        # Try to parse Title: / Source: from first two lines
        lines = content.splitlines()
        title = txt_file.stem.replace("_", " ").title()
        source = str(txt_file.name)
        if lines and lines[0].startswith("Title:"):
            title = lines[0][len("Title:"):].strip()
        if len(lines) > 1 and lines[1].startswith("Source:"):
            source = lines[1][len("Source:"):].strip()
        docs.append(Document(content=content, title=title, source=source))
    return docs


def interactive_mode(pipeline: RAGPipeline) -> None:
    """Run an interactive Q&A session."""
    print("\n" + "="*60)
    print("  RAG Interactive Mode")
    print("  Patterns: " + ", ".join(PATTERNS.keys()))
    print("  Type 'quit' to exit, 'help' for commands")
    print("="*60)

    current_pattern = "naive"

    while True:
        try:
            user_input = input(f"\n[{current_pattern}]> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user_input.lower() == "help":
            print("""
Commands:
  /pattern <name>    Switch to a different RAG pattern
  /compare           Run all patterns on next query
  /eval              Toggle metric evaluation on/off
  /patterns          List available patterns
  /chunks            Show number of indexed chunks
  quit               Exit
""")
            continue
        if user_input.lower() == "/patterns":
            for name, cls in PATTERNS.items():
                print(f"  {name:15s} — {cls.__module__.split('.')[-1]}")
            continue
        if user_input.lower() == "/chunks":
            print(f"  {pipeline.chunk_count} chunks indexed")
            continue
        if user_input.startswith("/pattern "):
            new_pattern = user_input[9:].strip()
            if new_pattern in PATTERNS:
                current_pattern = new_pattern
                print(f"  Switched to: {current_pattern}")
            else:
                print(f"  Unknown pattern. Available: {list(PATTERNS.keys())}")
            continue

        # Regular query
        print(f"\nRunning [{current_pattern}]…")
        try:
            result = pipeline.query(user_input, pattern=current_pattern)
            print("\n" + result.pretty())
        except Exception as e:
            print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Production RAG System with multiple retrieval strategies"
    )
    parser.add_argument("--query", "-q", type=str, help="Query to run")
    parser.add_argument(
        "--pattern", "-p",
        choices=list(PATTERNS.keys()),
        default="naive",
        help="RAG pattern to use (default: naive)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all patterns on the query and compare",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate faithfulness and relevance metrics",
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "sample_docs"),
        help="Directory containing .txt documents to index",
    )
    args = parser.parse_args()

    # Validate API key
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Load and index documents
    print(f"Loading documents from: {args.docs_dir}")
    docs = load_documents_from_dir(args.docs_dir)
    if not docs:
        print(f"No .txt files found in {args.docs_dir}", file=sys.stderr)
        sys.exit(1)

    pipeline = RAGPipeline()
    pipeline.index(docs)

    # Run query or interactive mode
    if args.query:
        if args.compare:
            print(f"\nRunning all patterns on: {args.query!r}\n")
            results = pipeline.compare(args.query)
            for pattern, result in results.items():
                print("=" * 60)
                print(result.pretty())
        elif args.evaluate:
            result = pipeline.query(args.query, pattern=args.pattern, evaluate=True)
            print(result.report())
        else:
            result = pipeline.query(args.query, pattern=args.pattern)
            print(result.pretty())
    else:
        interactive_mode(pipeline)


if __name__ == "__main__":
    main()
