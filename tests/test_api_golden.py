import pytest

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
        "evidence_anchor": "bring",
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
async def test_api_golden(asgi_client, case):
    resp = await asgi_client.post("/api/query", json={
        "question": case["question"],
        "stream": False,
        "debug": 1,
        "mode": "full",
        "top_k": 4,
        "conversation_id": None,
        "doc_ids": None,
    })
    assert resp.status_code == 200, resp.text
    data = resp.json()

    refused = data.get("refused", False)
    assert refused == case["expect_refused"], f"Refused mismatch for: {case['question']}"

    if not refused:
        debug = data.get("debug_info") or {}
        selected_chunks = debug.get("selected_chunks") or []
        assert selected_chunks, f"No selected_chunks for: {case['question']}"

        doc = (selected_chunks[0].get("doc") or "").lower()
        assert case["evidence_anchor"].lower() in doc, (
            f"Evidence anchor '{case['evidence_anchor']}' not in selected chunk doc for {case['question']}"
        )

        evidence = data.get("evidence") or []
        assert evidence, f"No evidence for: {case['question']}"

        # evidence objects are likely dicts or pydantic dicts; handle both
        first = evidence[0]
        snippet = (first.get("snippet") if isinstance(first, dict) else getattr(first, "snippet", "")) or ""
        assert case["evidence_anchor"].lower() in snippet.lower(), (
            f"Evidence anchor missing: {case['evidence_anchor']} for {case['question']}"
        )
