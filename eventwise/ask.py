"""Answer a question from the indexed documents.

    python -m eventwise.ask "question" [-k 10] [--doc NAME] [--no-vision] [--json] [--model MODEL]

Requires OPENAI_API_KEY, from the environment or from a .env file in the
current directory. Run from the project root so figure image paths and .env
resolve. One model call per question; there is no retry.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .answer import DEFAULT_MODEL, GroundedAnswer, answer_question
from .config import DEFAULT_INDEX_DIR
from .retrieve import HybridRetriever


def _print_answer(result: GroundedAnswer) -> None:
    print(result.answer)
    print(f"\nSufficient: {'yes' if result.sufficient else 'no'}")
    if result.citations:
        print("\nCitations:")
        for citation in result.citations:
            print(f"- {citation.citation}")
            print(f"  id={citation.element_id}")
    else:
        print("\nCitations: none")
    if result.images:
        print("\nFigures read: " + ", ".join(result.images))
    for warning in result.warnings:
        print(f"WARNING: {warning}")


def _load_local_env() -> None:
    """Load `.env` from the current directory. Variables already set are kept."""
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Answer a question from indexed documents.")
    parser.add_argument("query")
    parser.add_argument("-k", "--top-k", type=int, default=10)
    parser.add_argument("--doc", action="append", help="Restrict to documents (ID, name or substring); repeatable")
    parser.add_argument("--no-vision", action="store_true", help="Do not send figure images")
    parser.add_argument("--json", action="store_true", help="Print the structured answer as JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL})")
    parser.add_argument("--index", default=DEFAULT_INDEX_DIR)
    args = parser.parse_args(argv)
    _load_local_env()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        return 2

    from openai import OpenAI

    from .embeddings import SentenceTransformerEmbedder
    from .index import IndexStore, IndexUnavailable

    try:
        index = IndexStore(args.index).load()
    except IndexUnavailable as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    retriever = HybridRetriever(index, SentenceTransformerEmbedder(allow_download=False))
    try:
        result = answer_question(
            retriever, args.query, OpenAI(),
            top_k=args.top_k, documents=args.doc, vision=not args.no_vision, model=args.model,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=1))
    else:
        _print_answer(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
