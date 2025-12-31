import pytest
import json
from pathlib import Path
import os
import sys

# Ensure repo root is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from app.services.ingestion import load_file_to_text, chunk_text


@pytest.mark.parametrize("file_info", [
    pytest.param(
        {
            "file": "onboarding_guide.txt",
            "must_contain": [
                "8:00 AM",
                "9:00 AM",
                "12:00 PM",
                "3rd floor",
                "UNIQUE_TOKEN_ONBOARDING_7F3A9C"
            ]
        },
        id="onboarding_guide.txt"
    ),
    pytest.param(
        {
            "file": "facilities_parking.md",
            "must_contain": [
                "PARK-4421",
                "business casual",
                "UNIQUE_TOKEN_FACILITIES_2C91E0"
            ]
        },
        id="facilities_parking.md"
    ),
    pytest.param(
        {
            "file": "it_policy.txt",
            "must_contain": [
                "WIFI_PASSWORD: RAGIFY-1234",
                "RAGIFY-VPN",
                "UNIQUE_TOKEN_ITPOLICY_19D2B8"
            ]
        },
        id="it_policy.txt"
    ),
    pytest.param(
        {
            "file": "benefits_overview.txt",
            "must_contain": [
                "10 days",
                "after 30 days",
                "UNIQUE_TOKEN_BENEFITS_55AA01"
            ]
        },
        id="benefits_overview.txt"
    ),
    pytest.param(
        {
            "file": "employee_handbook_excerpt.pdf",
            "must_contain": [
                "10:30 AM",
                "Unique anchor: UNIQUE_TOKEN_PDF_88C0D1",
                "Unique anchor: UNIQUE_TOKEN_PDF_2_1A7B4F"
            ]
        },
        id="employee_handbook_excerpt.pdf"
    ),
    pytest.param(
        {
            "file": "onboarding_checklist.docx",
            "must_contain": [
                "reception",
                "9:00 AM",
                "Unique anchor: UNIQUE_TOKEN_DOCX_6F2D9A"
            ]
        },
        id="onboarding_checklist.docx"
    ),
    pytest.param(
        {
            "file": "edge_cases_chunking.txt",
            "must_contain": [
                "ANCHOR_SEQ_START",
                "ANCHOR_SEQ_END",
                "UNIQUE_TOKEN_EDGE_0B77CC"
            ]
        },
        id="edge_cases_chunking.txt"
    ),
])
def test_ingestion_and_chunking(file_info):
    """Test text extraction and chunking for each document type."""
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