#!/usr/bin/env python3
"""
Demo smoke test for RAGify MVP.

This script:
- Sets up CI environment variables
- Seeds vectorstore with test documents
- Tests 10 critical demo questions
- Prints compact results table
- Exits non-zero on failures
"""

import os
import sys
import asyncio
import shutil
import json
from pathlib import Path
from typing import Dict, List, Any

# Set CI environment before importing app modules
os.environ["APP_MODE"] = "ci"
os.environ["CI"] = "true"
os.environ["VECTOR_DIR"] = "vectorstore_test_smoke"
os.environ["ALLOW_CHROMA_INDEXING_IN_MOCK"] = "true"
os.environ["EMBEDDING_PROVIDER"] = "tfidf_test"

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.services import clients, rag_service
from app.services.ingestion import load_file_to_text, chunk_text

# Load standard questions from JSON file
STANDARD_QUESTIONS_FILE = REPO_ROOT / "tests" / "testdata" / "questions" / "standard_questions.json"
with open(STANDARD_QUESTIONS_FILE, 'r') as f:
    STANDARD_QUESTIONS = json.load(f)

# Filter to non-refused questions for demo
DEMO_QUESTIONS = [q for q in STANDARD_QUESTIONS if not q.get("expect_refused", False)]


async def seed_vectorstore():
    """Seed vectorstore with test documents."""
    print("🌱 Seeding vectorstore...")

    # Clean slate
    vector_dir = os.environ["VECTOR_DIR"]
    if os.path.exists(vector_dir):
        shutil.rmtree(vector_dir)
    os.makedirs(vector_dir, exist_ok=True)

    # Init Chroma
    clients.initialize_chroma_client()

    docs_dir = REPO_ROOT / "tests" / "testdata" / "docs"
    manifest_path = docs_dir / "manifest.json"

    # Load manifest
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_by_file = {m["file"]: m for m in manifest}

    # Deterministic ingestion order
    allowed_ext = {".txt", ".md", ".pdf", ".docx"}
    paths = sorted([p for p in docs_dir.iterdir() if p.suffix.lower() in allowed_ext])

    total_chunks = 0
    all_chunks = []  # Collect all chunks for TF-IDF fitting

    # First pass: collect all chunks
    for path in paths:
        filename = path.name
        print(f"  📄 Processing {filename}...")

        # Extract text
        text = load_file_to_text(str(path))

        # Verify manifest anchors exist
        spec = manifest_by_file.get(filename)
        if spec:
            for anchor in spec.get("must_contain", []):
                if anchor not in text:
                    print(f"  ❌ Missing anchor '{anchor}' in {filename}")
                    sys.exit(1)

        # Chunk and collect for TF-IDF fitting
        chunks = chunk_text(text)
        all_chunks.extend(chunks)

    # Fit TF-IDF embedder on all chunks BEFORE indexing
    if os.environ.get("EMBEDDING_PROVIDER") == "tfidf_test":
        print("🔧 Fitting TF-IDF test embedder...")
        rag_service.fit_tfidf_test_embedder(all_chunks)
        print("✅ TF-IDF test embedder fitted")

    # Second pass: index chunks
    for path in paths:
        filename = path.name

        # Extract text again
        text = load_file_to_text(str(path))
        chunks = chunk_text(text)

        # Index chunks
        n = await rag_service.add_documents("default", chunks, filename)
        total_chunks += n
        print(f"    ✅ Indexed {n} chunks from {filename}")

    print(f"🎯 Total chunks indexed: {total_chunks}")
    return total_chunks


async def run_demo_tests() -> List[Dict[str, Any]]:
    """Run demo questions and collect results."""
    print("\n🧪 Running demo tests...")

    results = []

    for i, case in enumerate(DEMO_QUESTIONS, 1):
        question = case["question"]
        print(f"  {i:2d}. {question}")

        try:
            # Query collection
            answer_gen, sources, evidence_items, context, debug_info = await rag_service.query_collection(
                tenant_id="default",
                question=question,
                top_k=4,
                debug=1,
                request_id=f"demo-{i}"
            )

            # Extract results
            refused = debug_info.get("refused", False)
            selected_chunks = debug_info.get("selected_chunks", [])

            # Get top selected header
            top_header = ""
            if selected_chunks:
                top_header = selected_chunks[0].get("header_first_line", "")

            # Get top source file
            top_source = ""
            if selected_chunks:
                top_source = selected_chunks[0].get("source_file", "")

            # Check if expected anchor found in selected chunks
            anchor_found = False
            if not refused and selected_chunks:
                for chunk in selected_chunks:
                    chunk_doc = chunk.get("doc", "")
                    if case["expected_anchor"] in chunk_doc:
                        anchor_found = True
                        break

            # Check if expected source file is in selected chunks
            source_found = False
            if not refused and selected_chunks and case.get("expected_file"):
                for chunk in selected_chunks:
                    if chunk.get("source_file") == case["expected_file"]:
                        source_found = True
                        break

            result = {
                "question": question[:50] + "..." if len(question) > 50 else question,
                "refused": refused,
                "top_selected_header": top_header[:30] + "..." if len(top_header) > 30 else top_header,
                "evidence_anchor_found": anchor_found,
                "source_file_found": source_found,
                "top_source_file": top_source,
                "expected_refused": case["expect_refused"],
                "expected_anchor": case["expected_anchor"],
                "expected_file": case.get("expected_file", ""),
            }

            results.append(result)

        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({
                "question": question[:50] + "..." if len(question) > 50 else question,
                "refused": True,
                "top_selected_header": "ERROR",
                "evidence_anchor_found": False,
                "source_file_found": False,
                "top_source_file": "ERROR",
                "expected_refused": case["expect_refused"],
                "expected_anchor": case["expected_anchor"],
                "expected_file": case.get("expected_file", ""),
                "error": str(e),
            })

    return results


def print_results_table(results: List[Dict[str, Any]]):
    """Print compact results table."""
    print("\n📊 Demo Smoke Test Results")
    print("=" * 140)

    # Header
    print(f"{'#':<2} {'Question':<50} {'Refused':<7} {'Header':<30} {'Anchor':<6} {'Source':<6} {'File':<20}")
    print("-" * 140)

    # Rows
    failures = 0
    for i, result in enumerate(results, 1):
        q = result["question"]
        refused = "YES" if result["refused"] else "NO"
        header = result["top_selected_header"]
        anchor = "YES" if result["evidence_anchor_found"] else "NO"
        source_match = "YES" if result["source_file_found"] else "NO"
        file = result["top_source_file"]

        # Determine if this is a failure
        expected_refused = result["expected_refused"]
        is_failure = False
        
        if expected_refused:
            # Should be refused
            is_failure = not result["refused"]
        else:
            # Should not be refused and should find anchor and source
            is_failure = (result["refused"] or 
                         not result["evidence_anchor_found"] or 
                         not result["source_file_found"])
        
        if is_failure:
            failures += 1
            # Color coding for failures (using ANSI escape codes)
            print(f"❌{i:<2d} {q:<50} {refused:<7} {header:<30} {anchor:<6} {source_match:<6} {file:<20}")
        else:
            print(f"✅{i:<2d} {q:<50} {refused:<7} {header:<30} {anchor:<6} {source_match:<6} {file:<20}")

    print("=" * 140)
    print(f"Total: {len(results)}, Failures: {failures}")

    return failures

    return failures


async def main():
    """Main demo smoke test."""
    print("🚀 RAGify Demo Smoke Test")
    print(f"Chunk size: 300, Overlap: 50 (demo mode)")

    try:
        # Seed vectorstore
        total_chunks = await seed_vectorstore()
        if total_chunks == 0:
            print("❌ No chunks indexed!")
            sys.exit(1)

        # Run tests
        results = await run_demo_tests()

        # Print results
        failures = print_results_table(results)

        # Exit with failure code if any tests failed
        if failures > 0:
            print(f"\n❌ {failures} test(s) failed!")
            sys.exit(1)
        else:
            print("\n✅ All tests passed!")
            sys.exit(0)

    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())