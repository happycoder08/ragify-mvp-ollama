import pytest
import json
from pathlib import Path
import os
import sys

# Ensure repo root is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from app.services.ingestion import load_file_to_text, chunk_text


def test_ingestion_and_chunking(standard_questions):
    """Test text extraction and chunking for each document type using standard questions."""
    # Group questions by expected file to collect anchors
    file_anchors = {}
    for question in standard_questions:
        if question.get("expected_file") and question.get("expected_anchor"):
            filename = question["expected_file"]
            if filename not in file_anchors:
                file_anchors[filename] = []
            file_anchors[filename].append(question["expected_anchor"])

    # Convert to test parameters
    test_cases = []
    for filename, anchors in file_anchors.items():
        test_cases.append({
            "file": filename,
            "must_contain": anchors
        })

    for file_info in test_cases:
        filename = file_info["file"]
        must_contain = file_info["must_contain"]

        # Construct file path
        docs_dir = Path(REPO_ROOT) / "tests" / "testdata" / "docs"
        file_path = docs_dir / filename

        # Ensure file exists
        assert file_path.exists(), f"Test file {filename} does not exist at {file_path}"

        # Test text extraction
        text = load_file_to_text(str(file_path))

        # Assert text is substantial
        assert len(text) > 200, f"Extracted text from {filename} is too short: {len(text)} chars"

        # Assert all required tokens are present in extracted text
        for token in must_contain:
            assert token in text, f"Required token '{token}' not found in extracted text from {filename}"

        # Test chunking
        chunks = chunk_text(text)

        # Assert reasonable chunk count
        assert 1 <= len(chunks) <= 5000, f"Chunking produced {len(chunks)} chunks for {filename}, expected 1-5000"

        # Assert all required tokens appear in at least one chunk
        for token in must_contain:
            chunk_contains_token = any(token in chunk for chunk in chunks)
            assert chunk_contains_token, f"Required token '{token}' not found in any chunk from {filename}"

        # Report chunk count in test output
        print(f"{filename}: {len(chunks)} chunks generated")