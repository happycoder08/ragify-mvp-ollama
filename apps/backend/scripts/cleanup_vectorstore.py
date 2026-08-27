#!/usr/bin/env python3
"""
Cleanup script for RAGify vector store.

Removes all ChromaDB files and indexed documents to reset the system.
Useful for:
  - Clearing old/stale embeddings
  - Starting fresh with new documents
  - Freeing up disk space
  - Debugging indexing issues

Usage:
  python scripts/cleanup_vectorstore.py
"""

import os
import shutil
import sys

from app.config import VECTOR_DIR, UPLOAD_DIR


def cleanup_vectorstore():
    """Remove all ChromaDB persisted files."""
    if not os.path.exists(VECTOR_DIR):
        print(f"✓ Vector store directory does not exist: {VECTOR_DIR}")
        return True

    try:
        print(f"Removing vector store at: {VECTOR_DIR}")
        shutil.rmtree(VECTOR_DIR)
        print(f"✓ Vector store cleaned successfully")
        return True
    except Exception as e:
        print(f"✗ Error removing vector store: {e}")
        return False


def cleanup_uploads():
    """Remove all uploaded documents."""
    if not os.path.exists(UPLOAD_DIR):
        print(f"✓ Upload directory does not exist: {UPLOAD_DIR}")
        return True

    try:
        print(f"Removing uploaded files at: {UPLOAD_DIR}")
        shutil.rmtree(UPLOAD_DIR)
        print(f"✓ Upload directory cleaned successfully")
        return True
    except Exception as e:
        print(f"✗ Error removing upload directory: {e}")
        return False


def main():
    """Main cleanup routine."""
    print("=" * 60)
    print("RAGify Vector Store & Upload Cleanup")
    print("=" * 60)
    print()

    # Ask for confirmation
    response = input("This will delete all indexed embeddings and uploaded files. Continue? (y/n): ").strip().lower()
    if response != "y":
        print("Cleanup cancelled.")
        sys.exit(0)

    print()
    success = True
    success &= cleanup_vectorstore()
    success &= cleanup_uploads()

    print()
    print("=" * 60)
    if success:
        print("✓ Cleanup completed successfully")
        print()
        print("Next steps:")
        print("  1. Start the server: uvicorn main:app --reload")
        print("  2. Upload new documents via the web interface")
        print("  3. Ask questions to build a fresh index")
        sys.exit(0)
    else:
        print("✗ Cleanup completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
