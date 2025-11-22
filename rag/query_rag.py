#!/usr/bin/env python3
"""
Query Legal RAG System

Interactive CLI for querying the Legal RAG system with multiple LLM providers.
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# Try to load from cross-llm-mcp/.env first, then local .env
cross_llm_env = Path("/Users/jamessangalli/Documents/projects/cross-llm-mcp/.env")
if cross_llm_env.exists():
    load_dotenv(cross_llm_env)
load_dotenv()  # Also load from local .env if exists (local overrides)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rag_system import LegalRAG
from rag.config import (
    SUPPORTED_LLMS,
    LLM_PROVIDERS,
    VECTORSTORE_PATH,
    DATA_DIR,
    DEFAULT_TOP_K,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    validate_api_key,
)


def print_sources(result: dict, show_full_content: bool = False) -> None:
    """Print source documents from query result."""
    sources = result.get("source_documents", [])
    if sources:
        print("\n" + "=" * 80)
        print("Source Documents:")
        print("=" * 80)
        for i, source in enumerate(sources, 1):
            metadata = source.get("metadata", {})
            filename = metadata.get("source", "Unknown")
            # Extract just the filename
            if isinstance(filename, str):
                filename = Path(filename).name

            content = source.get("content", "")
            if not show_full_content and len(content) > 500:
                content = content[:500] + "..."

            print(f"\n[{i}] Source: {filename}")
            print(f"Content: {content}")
        print("=" * 80)


def interactive_query(rag: LegalRAG, args: argparse.Namespace) -> None:
    """Run interactive query loop."""
    print("\n" + "=" * 80)
    print("Legal RAG Query Interface")
    print("=" * 80)
    print(f"LLM Provider: {args.llm}")
    print(f"Model: {args.model or LLM_PROVIDERS[args.llm].default_model}")
    print("Enter your legal questions (type 'quit' or 'exit' to stop)")
    print("Commands: 'quit', 'exit', 'q' - quit")
    print("         'llm <provider>' - switch LLM provider")
    print("         'model <model>' - change model")
    print()

    current_llm = args.llm
    current_model = args.model
    current_api_key = args.api_key

    while True:
        try:
            question = input("Question: ").strip()

            if question.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            if not question:
                continue

            # Handle commands
            if question.lower().startswith("llm "):
                new_llm = question.split(" ", 1)[1].strip().lower()
                if new_llm in SUPPORTED_LLMS:
                    current_llm = new_llm
                    current_model = None  # Reset to default
                    print(f"Switched to LLM provider: {current_llm}")
                    print(f"Default model: {LLM_PROVIDERS[current_llm].default_model}")
                else:
                    print(
                        f"Invalid LLM provider. Supported: {', '.join(SUPPORTED_LLMS)}"
                    )
                continue

            if question.lower().startswith("model "):
                new_model = question.split(" ", 1)[1].strip()
                current_model = new_model
                print(f"Using model: {current_model}")
                continue

            # Validate API key
            try:
                validate_api_key(current_llm, current_api_key)
            except ValueError as e:
                print(f"\nError: {e}")
                print("Please set the API key via --api-key or environment variable")
                continue

            print("\nGenerating answer...")
            try:
                result = rag.query(
                    question=question,
                    provider=current_llm,
                    model=current_model,
                    api_key=current_api_key,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_k=args.top_k,
                )

                print(f"\nAnswer: {result['answer']}")

                if args.show_sources:
                    print_sources(result, show_full_content=args.show_full_content)

            except ValueError as e:
                print(f"\nError: {e}\n")
            except Exception as e:
                print(f"\nUnexpected error: {e}\n")
                import traceback

                traceback.print_exc()

            print("-" * 80)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Query the Legal RAG system with multiple LLM providers"
    )
    parser.add_argument(
        "--llm",
        type=str,
        choices=SUPPORTED_LLMS,
        default="chatgpt",
        help=f'LLM provider to use (default: chatgpt). Options: {", ".join(SUPPORTED_LLMS)}',
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model to use (uses provider default if not specified)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the LLM provider (overrides environment variable)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question to ask (if not provided, runs interactive mode)",
    )
    parser.add_argument(
        "--vectorstore-path",
        type=str,
        default=VECTORSTORE_PATH,
        help=f"Path to vector store directory (default: {VECTORSTORE_PATH})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens in response (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of documents to retrieve (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--show-sources", action="store_true", help="Show source documents with answers"
    )
    parser.add_argument(
        "--show-full-content",
        action="store_true",
        help="Show full content of source documents (use with --show-sources)",
    )

    args = parser.parse_args()

    # Check if vector store exists
    vectorstore_path = Path(args.vectorstore_path)
    if not vectorstore_path.exists() or not list(vectorstore_path.glob("*.faiss")):
        print(f"ERROR: Vector store not found at {args.vectorstore_path}")
        print("  Please run build_index.py first to build the vector store")
        sys.exit(1)

    # Validate API key
    try:
        validate_api_key(args.llm, args.api_key)
    except ValueError as e:
        print(f"ERROR: {e}")
        print(f"\nTo get an API key:")
        config = LLM_PROVIDERS[args.llm]
        print(f"  1. Visit the {config.name} website")
        print(f"  2. Create an API key")
        print(f"  3. Set environment variable {config.env_var} or use --api-key")
        sys.exit(1)

    # Initialize RAG system
    print(f"Loading vector store from {args.vectorstore_path}...")
    try:
        rag = LegalRAG(vectorstore_path=str(vectorstore_path))
        rag.build_index(force_rebuild=False)
    except Exception as e:
        print(f"ERROR: Failed to load vector store: {e}")
        sys.exit(1)

    # Single question or interactive mode
    if args.question:
        try:
            result = rag.query(
                question=args.question,
                provider=args.llm,
                model=args.model,
                api_key=args.api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_k=args.top_k,
            )

            print(f"\nQuestion: {args.question}")
            print(f"Answer: {result['answer']}")

            if args.show_sources:
                print_sources(result, show_full_content=args.show_full_content)

        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
    else:
        interactive_query(rag, args)


if __name__ == "__main__":
    main()
