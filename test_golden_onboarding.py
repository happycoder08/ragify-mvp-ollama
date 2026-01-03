
import pytest
import asyncio
from app.services.rag_service import query_collection
import sys


# Ensure HTTP client is initialized for embedding provider (mock or real)
import pytest
@pytest.fixture(autouse=True, scope="session")
async def init_http_client():
    try:
        from app.services import clients
        if hasattr(clients, "initialize_http_client"):
            await clients.initialize_http_client()
        # Teardown: close client if available
        yield
        if hasattr(clients, "close_http_client"):
            close_fn = clients.close_http_client
            if asyncio.iscoroutinefunction(close_fn):
                await close_fn()
            else:
                close_fn()
    except Exception as e:
        import sys
        print(f"[WARN] Could not initialize/close HTTP client: {e}", file=sys.stderr)

@pytest.fixture(autouse=True, scope="session")
async def init_chroma_client():
    try:
        from app.services import clients
        if hasattr(clients, "initialize_chroma_client"):
            clients.initialize_chroma_client()
        # No teardown needed for ChromaDB
        yield
    except Exception as e:
        import sys
        print(f"[WARN] Could not initialize ChromaDB client: {e}", file=sys.stderr)

# Golden set: question, expected header keyword, expected evidence anchor, expect_refused
GOLDEN_SET = [
    {
        "question": "What time do I arrive my first day?",
        "header_kw": ["arrive", "morning"],
        "evidence_anchor": "8:00",
        "expect_refused": False,
    },
    {
        "question": "When is team lunch?",
        "header_kw": ["lunch"],
        "evidence_anchor": "12:00",
        "expect_refused": False,
    },
    {
        "question": "Where is the main reception?",
        "header_kw": ["reception"],
        "evidence_anchor": "reception",
        "expect_refused": False,
    },
    {
        "question": "Who is my manager?",
        "header_kw": ["manager"],
        "evidence_anchor": "manager",
        "expect_refused": False,
    },
    {
        "question": "What is the wifi password?",
        "header_kw": ["wifi"],
        "evidence_anchor": "wifi",
        "expect_refused": False,
    },
    {
        "question": "How do I set up my email signature?",
        "header_kw": ["signature", "email"],
        "evidence_anchor": "signature",
        "expect_refused": False,
    },
    {
        "question": "When do I get my badge?",
        "header_kw": ["badge"],
        "evidence_anchor": "badge",
        "expect_refused": False,
    },
    {
        "question": "What documents do I need to bring?",
        "header_kw": ["document", "bring"],
        "evidence_anchor": "document",
        "expect_refused": False,
    },
    {
        "question": "Is there a dress code?",
        "header_kw": ["dress"],
        "evidence_anchor": "dress",
        "expect_refused": False,
    },
    {
        "question": "What time does orientation start?",
        "header_kw": ["orientation", "start"],
        "evidence_anchor": "9:00",
        "expect_refused": False,
    },
]

@pytest.mark.asyncio
@pytest.mark.parametrize("case", GOLDEN_SET)
async def test_golden_onboarding(case):
    # Use tenant_id="default" and debug=1 for maximal info
    result = await query_collection(
        tenant_id="default",
        question=case["question"],
        top_k=4,
        debug=1,
        request_id="golden-test"
    )
    answer_gen, source_files, evidence_items, context_text, debug_info = result
    # Check debug_info for selected_chunks
    selected_chunks = debug_info.get("selected_chunks") or []
    refused = debug_info.get("refused", False)
    # If refused, must match expectation
    assert refused == case["expect_refused"], f"Refused mismatch for: {case['question']}"
    if not refused:
        # Check header keyword in selected_chunks[0]
        header = selected_chunks[0]["header_first_line"].lower() if selected_chunks else ""
        assert any(kw in header for kw in case["header_kw"]), f"Header mismatch: {header} for {case['question']}"
        # Check evidence anchor in evidence_items[0].snippet
        snippet = evidence_items[0].snippet.lower() if evidence_items else ""
        assert case["evidence_anchor"].lower() in snippet, f"Evidence anchor missing: {case['evidence_anchor']} for {case['question']}"
