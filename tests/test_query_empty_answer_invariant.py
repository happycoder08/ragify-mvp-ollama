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
    assert payload["answer"].startswith("Employees should arrive at 9:00 AM.")
    assert payload["pipeline_marker"] == "VALIDATION_FAILED_FALLBACK"
    assert payload["debug_info"]["empty_answer_invariant_tripped"] is True
    assert payload["debug_info"]["fallback_from_evidence"] is True
    assert payload["evidence"]
    assert payload["sources"]


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
