import pytest

import main
from app.schemas.query import AnswerDecision, DecisionType, AnswerSchema, EvidenceItem


@pytest.mark.asyncio
async def test_empty_answer_fallback_from_evidence(asgi_client, monkeypatch):
    async def mock_answer_question(*args, **kwargs):
        async def gen():
            if False:
                yield ""

        evidence = [
            EvidenceItem(
                snippet="Employees should arrive at 9:00 AM.",
                chunk_id="chunk_1",
                heading="Arrival Time",
                doc_id=1,
            )
        ]
        sources = ["onboarding.txt#chunk_1"]
        debug_payload = {}
        decision = AnswerDecision(
            decision_type=DecisionType.LLM_VALIDATED,
            refused=False,
            refusal_reason=None,
            answer_schema=AnswerSchema.FACT_SINGLE,
        )
        return gen(), sources, evidence, "", debug_payload, decision

    monkeypatch.setattr(main, "answer_question", mock_answer_question)

    response = await asgi_client.post(
        "/api/query",
        json={"question": "When should I arrive?", "stream": False, "debug": 1},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["refused"] is False
    assert payload["answer"].strip() != ""
    assert "9:00 AM" in payload["answer"]
    assert payload["pipeline_marker"].startswith("EXTRACTOR_")
    assert payload["debug_info"]["empty_answer_invariant_tripped"] is True
    assert payload["debug_info"]["fallback_from_evidence"] is True
    assert payload["evidence"]
    assert payload["sources"]


@pytest.mark.asyncio
async def test_empty_answer_fallback_fact_single_vacation_days(asgi_client, monkeypatch):
    async def mock_answer_question(*args, **kwargs):
        async def gen():
            if False:
                yield ""

        evidence = [
            EvidenceItem(
                snippet="Full-time employees receive 15 vacation days per year.",
                chunk_id="chunk_1",
                heading="Vacation Policy",
                doc_id=1,
            )
        ]
        sources = ["policy.txt#chunk_1"]
        debug_payload = {}
        decision = AnswerDecision(
            decision_type=DecisionType.LLM_VALIDATED,
            refused=False,
            refusal_reason=None,
            answer_schema=AnswerSchema.FACT_SINGLE,
        )
        return gen(), sources, evidence, "", debug_payload, decision

    monkeypatch.setattr(main, "answer_question", mock_answer_question)

    response = await asgi_client.post(
        "/api/query",
        json={"question": "How many vacation days are there in 2025?", "stream": False, "debug": 1},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["refused"] is False
    assert payload["answer"] == "Full-time employees receive 15 vacation days per year."
    assert payload["pipeline_marker"] == "EXTRACTOR_FACT_SINGLE"
    assert payload["debug_info"]["empty_answer_invariant_tripped"] is True
    assert payload["debug_info"]["fallback_from_evidence"] is True


@pytest.mark.asyncio
async def test_empty_answer_fallback_schema_missing_fact_single(asgi_client, monkeypatch):
    async def mock_answer_question(*args, **kwargs):
        async def gen():
            if False:
                yield ""

        evidence = [
            EvidenceItem(
                snippet="Full-time employees receive 15 vacation days per year.",
                chunk_id="chunk_1",
                heading="Vacation Policy",
                doc_id=1,
            )
        ]
        sources = ["policy.txt#chunk_1"]
        debug_payload = {}
        decision = AnswerDecision(
            decision_type=DecisionType.LLM_VALIDATED,
            refused=False,
            refusal_reason=None,
            answer_schema=None,
        )
        return gen(), sources, evidence, "", debug_payload, decision

    monkeypatch.setattr(main, "answer_question", mock_answer_question)

    response = await asgi_client.post(
        "/api/query",
        json={"question": "How many vacation days are there in 2025?", "stream": False, "debug": 1},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["refused"] is False
    assert payload["answer"] == "Full-time employees receive 15 vacation days per year."
    assert payload["pipeline_marker"] == "EXTRACTOR_FACT_SINGLE"
    assert payload["debug_info"]["empty_answer_invariant_tripped"] is True
    assert payload["debug_info"]["fallback_from_evidence"] is True


@pytest.mark.asyncio
async def test_empty_answer_fallback_prefers_year_2025_vacation(asgi_client, monkeypatch):
    async def mock_answer_question(*args, **kwargs):
        async def gen():
            if False:
                yield ""

        evidence = [
            EvidenceItem(
                snippet="BENEFITS POLICY (2026)\nVACATION\nFull-time employees receive 20 vacation days per year.",
                chunk_id="chunk_2026",
                heading="BENEFITS POLICY (2026)",
                doc_id=2,
            ),
            EvidenceItem(
                snippet="BENEFITS POLICY (2025)\nVACATION\nFull-time employees receive 15 vacation days per year.",
                chunk_id="chunk_2025",
                heading="BENEFITS POLICY (2025)",
                doc_id=1,
            ),
        ]
        sources = ["benefits-2026.txt#chunk_2026", "benefits-2025.txt#chunk_2025"]
        debug_payload = {}
        decision = AnswerDecision(
            decision_type=DecisionType.LLM_VALIDATED,
            refused=False,
            refusal_reason=None,
            answer_schema=AnswerSchema.SUMMARY_OVERVIEW,
        )
        return gen(), sources, evidence, "", debug_payload, decision

    monkeypatch.setattr(main, "answer_question", mock_answer_question)

    response = await asgi_client.post(
        "/api/query",
        json={"question": "How many vacation days are there in 2025?", "stream": False, "debug": 1},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["refused"] is False
    assert "vacation" in payload["answer"].lower()
    assert "15" in payload["answer"]
    assert "BENEFITS POLICY" not in payload["answer"]
    assert payload["pipeline_marker"] == "EXTRACTOR_FACT_SINGLE"


@pytest.mark.asyncio
async def test_empty_answer_fallback_prefers_year_2026_vacation(asgi_client, monkeypatch):
    async def mock_answer_question(*args, **kwargs):
        async def gen():
            if False:
                yield ""

        evidence = [
            EvidenceItem(
                snippet="BENEFITS POLICY (2025)\nVACATION\nFull-time employees receive 15 vacation days per year.",
                chunk_id="chunk_2025",
                heading="BENEFITS POLICY (2025)",
                doc_id=1,
            ),
            EvidenceItem(
                snippet="BENEFITS POLICY (2026)\nVACATION\nFull-time employees receive 20 vacation days per year.",
                chunk_id="chunk_2026",
                heading="BENEFITS POLICY (2026)",
                doc_id=2,
            ),
        ]
        sources = ["benefits-2025.txt#chunk_2025", "benefits-2026.txt#chunk_2026"]
        debug_payload = {}
        decision = AnswerDecision(
            decision_type=DecisionType.LLM_VALIDATED,
            refused=False,
            refusal_reason=None,
            answer_schema=AnswerSchema.SUMMARY_OVERVIEW,
        )
        return gen(), sources, evidence, "", debug_payload, decision

    monkeypatch.setattr(main, "answer_question", mock_answer_question)

    response = await asgi_client.post(
        "/api/query",
        json={"question": "How many vacation days are there in 2026?", "stream": False, "debug": 1},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["refused"] is False
    assert "vacation" in payload["answer"].lower()
    assert "20" in payload["answer"]
    assert "BENEFITS POLICY" not in payload["answer"]
    assert payload["pipeline_marker"] == "EXTRACTOR_FACT_SINGLE"


@pytest.mark.asyncio
async def test_clarification_response_answer_non_empty(asgi_client, monkeypatch):
    async def mock_answer_question(*args, **kwargs):
        async def gen():
            if False:
                yield ""

        debug_payload = {
            "pipeline_marker": "CLARIFICATION_REQUIRED",
            "clarification": {
                "type": "TIMEFRAME",
                "question": "Which year are you referring to?",
                "options": ["2025", "2026"],
            },
        }
        decision = AnswerDecision(
            decision_type=DecisionType.LLM_VALIDATED,
            refused=False,
            refusal_reason=None,
        )
        return gen(), [], [], "", debug_payload, decision

    monkeypatch.setattr(main, "answer_question", mock_answer_question)

    response = await asgi_client.post(
        "/api/query",
        json={"question": "How many vacation days?", "stream": False, "debug": 0},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["pipeline_marker"] == "CLARIFICATION_REQUIRED"
    assert payload["needs_clarification"] is True
    assert payload["answer"].strip() == "Which year are you referring to?"
