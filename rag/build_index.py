#!/usr/bin/env python3
"""
Build Vector Store Index for Legal RAG System

Loads legal documents from data/ directory, applies cleaning,
splits into chunks, generates embeddings, and builds FAISS vector store.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.rag_system import LegalRAG
from rag.config import VECTORSTORE_PATH, DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def main():
    """Build or rebuild the vector store index."""
    parser = argparse.ArgumentParser(
        description="Build vector store index for Legal RAG system"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR,
        help=f"Path to data directory with legal documents (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--vectorstore-path",
        type=str,
        default=VECTORSTORE_PATH,
        help=f"Path to vector store directory (default: {VECTORSTORE_PATH})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Size of text chunks (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Overlap between chunks (default: {CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuild of index even if it exists",
    )

    args = parser.parse_args()

    # Check if data directory exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"ERROR: Data directory not found: {args.data_dir}")
        print(
            "  Please ensure the data directory exists with legal document .txt files"
        )
        sys.exit(1)

    # Count .txt files
    txt_files = list(data_path.glob("**/*.txt"))
    if not txt_files:
        print(f"ERROR: No .txt files found in {args.data_dir}")
        sys.exit(1)

    print("=" * 80)
    print("Legal RAG - Build Vector Store Index")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    num_files = len(txt_files)
    print(f"Found {num_files} .txt files")
    print(f"Vector store path: {args.vectorstore_path}")
    print(f"Chunk size: {args.chunk_size}, overlap: {args.chunk_overlap}")
    print()

    # Initialize RAG system
    try:
        rag = LegalRAG(
            vectorstore_path=args.vectorstore_path,
            data_dir=args.data_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )

        # Build index
        rag.build_index(force_rebuild=args.force_rebuild)

        print()
        print("=" * 80)
        print("Index build complete!")
        print("=" * 80)
        print(f"Vector store saved to: {args.vectorstore_path}")
        print("You can now use query_rag.py to query the RAG system")
        print()

    except Exception as e:
        print(f"\nERROR: Failed to build index: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
