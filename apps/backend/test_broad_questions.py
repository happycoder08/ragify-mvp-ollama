import pytest
from app.services import rag_service


class BroadQuestionFakeCollection:
    def count(self):
        return 6

    def query(self, query_embeddings, n_results, where=None, include=None):
        # Mock data for broad questions with multiple headings and checklist items
        return {
            "documents": [[
                "# First Day Checklist\n\nArrive at 8:00 AM at reception. Check in with HR. Get your badge. Attend orientation at 9:00 AM.",
                "# IT Setup\n\nSet up your email and computer. Connect to wifi network 'CompanyNet'.",
                "# Lunch Schedule\n\nTeam lunch at 12:00 PM in the cafeteria. Meet your manager.",
                "# Health Insurance\n\nCoverage starts on your first day. Benefits enrollment due within 30 days.",
                "# Office Location\n\nMain reception on 3rd floor. Your desk is in room 305.",
                "# Emergency Procedures\n\nFire exits located near elevators. Emergency number is 911.",
            ]],
            "metadatas": [[
                {"source_file": "onboarding.txt", "chunk": 0, "header": "First Day Checklist"},
                {"source_file": "onboarding.txt", "chunk": 1, "header": "IT Setup"},
                {"source_file": "onboarding.txt", "chunk": 2, "header": "Lunch Schedule"},
                {"source_file": "onboarding.txt", "chunk": 3, "header": "Health Insurance"},
                {"source_file": "onboarding.txt", "chunk": 4, "header": "Office Location"},
                {"source_file": "onboarding.txt", "chunk": 5, "header": "Emergency Procedures"},
            ]],
            "distances": [[50.0, 60.0, 70.0, 80.0, 90.0, 100.0]],
            "ids": [["doc1_0", "doc1_1", "doc1_2", "doc1_3", "doc1_4", "doc1_5"]],
            "embeddings": [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8]]],
        }


@pytest.mark.asyncio
async def test_broad_question_first_day_checklist(monkeypatch):
    """Test broad question: 'What do I do on my first day?'"""
    
    async def fake_embed_texts(texts, tenant_id="default"):
        return [[0.0] * 3 for _ in texts]

    async def fake_get_collection(tenant_id: str):
        return BroadQuestionFakeCollection()

    def fake_call_chat_model(question, context, tenant_id, mode, conversation_history, request_id, prompt_template):
        # Mock checklist response for broad question
        async def gen():
            yield """1. According to First Day Checklist: Arrive at 8:00 AM at reception
2. According to First Day Checklist: Check in with HR
3. According to IT Setup: Set up your email and computer
4. According to IT Setup: Connect to wifi network 'CompanyNet'
5. According to Lunch Schedule: Team lunch at 12:00 PM in the cafeteria"""
        return gen()

    monkeypatch.setattr(rag_service, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(rag_service, "get_collection_async", fake_get_collection)
    monkeypatch.setattr(rag_service, "_call_chat_model", fake_call_chat_model)

    answer_gen, sources, evidence, context, debug_info = await rag_service.query_collection(
        tenant_id="test-tenant",
        question="What do I do on my first day?",
        top_k=6,
        debug=1,
        request_id="broad-test-1",
    )

    # Collect answer
    answer_text = ""
    async for chunk in answer_gen:
        answer_text += chunk

    # Assertions for broad question
    selected_chunks = debug_info.get("selected_chunks") or []
    
    # Check distinct headings (>= 3)
    headings = set()
    for chunk in selected_chunks:
        header = chunk.get("header_first_line", "").strip().lower()
        if header:
            headings.add(header)
    assert len(headings) >= 3, f"Expected at least 3 distinct headings, got {len(headings)}"
    
    # Check checklist keywords in evidence
    evidence_text = " ".join([item.snippet.lower() for item in evidence if hasattr(item, 'snippet')])
    checklist_keywords = ["arrive", "hr", "it", "lunch", "checklist"]
    found_keywords = [kw for kw in checklist_keywords if kw.lower() in evidence_text]
    assert len(found_keywords) > 0, f"No checklist keywords found in evidence: {checklist_keywords}"
    
    # Check debug info
    assert debug_info["is_broad"] is True
    assert debug_info["selected_by"] == "mmr"


@pytest.mark.asyncio
async def test_broad_question_arrival_time(monkeypatch):
    """Test broad question: 'What time should I arrive on my first day?'"""
    
    async def fake_embed_texts(texts, tenant_id="default"):
        return [[0.0] * 3 for _ in texts]

    async def fake_get_collection(tenant_id: str):
        return BroadQuestionFakeCollection()

    def fake_call_chat_model(question, context, tenant_id, mode, conversation_history, request_id, prompt_template):
        async def gen():
            yield "According to First Day Checklist: Arrive at 8:00 AM at reception"
        return gen()

    monkeypatch.setattr(rag_service, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(rag_service, "get_collection_async", fake_get_collection)
    monkeypatch.setattr(rag_service, "_call_chat_model", fake_call_chat_model)

    answer_gen, sources, evidence, context, debug_info = await rag_service.query_collection(
        tenant_id="test-tenant",
        question="What time should I arrive on my first day?",
        top_k=6,
        debug=1,
        request_id="broad-test-2",
    )

    # Collect answer
    answer_text = ""
    async for chunk in answer_gen:
        answer_text += chunk

    # Check answer contains time keywords
    answer_lower = answer_text.lower()
    time_keywords = ["8:00", "eight"]
    found_keywords = [kw for kw in time_keywords if kw.lower() in answer_lower]
    assert len(found_keywords) > 0, f"Expected time keywords not found: {time_keywords}, Answer: {answer_text[:200]}"
    
    # Check evidence contains arrival info
    evidence_text = " ".join([item.snippet.lower() for item in evidence if hasattr(item, 'snippet')])
    assert "arrive" in evidence_text, "Evidence should contain arrival information"
    
    assert debug_info["is_broad"] is True


@pytest.mark.asyncio
async def test_broad_question_health_insurance(monkeypatch):
    """Test broad question: 'When does health insurance coverage start?'"""
    
    async def fake_embed_texts(texts, tenant_id="default"):
        return [[0.0] * 3 for _ in texts]

    async def fake_get_collection(tenant_id: str):
        return BroadQuestionFakeCollection()

    def fake_call_chat_model(question, context, tenant_id, mode, conversation_history, request_id, prompt_template):
        async def gen():
            yield "According to Health Insurance: Coverage starts on your first day"
        return gen()

    monkeypatch.setattr(rag_service, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(rag_service, "get_collection_async", fake_get_collection)
    monkeypatch.setattr(rag_service, "_call_chat_model", fake_call_chat_model)

    answer_gen, sources, evidence, context, debug_info = await rag_service.query_collection(
        tenant_id="test-tenant",
        question="When does health insurance coverage start?",
        top_k=6,
        debug=1,
        request_id="broad-test-3",
    )

    # Collect answer
    answer_text = ""
    async for chunk in answer_gen:
        answer_text += chunk

    # Check answer mentions first day
    answer_lower = answer_text.lower()
    assert "first day" in answer_lower or "day one" in answer_lower, f"Answer should mention first day: {answer_text[:200]}"
    
    # Check evidence is from health insurance section
    evidence_text = " ".join([item.snippet.lower() for item in evidence if hasattr(item, 'snippet')])
    assert "health insurance" in evidence_text, "Evidence should be from health insurance section"
    
    assert debug_info["is_broad"] is True