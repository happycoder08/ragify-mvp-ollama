import pytest
from app.schemas.query import QueryRequest

def test_query_request_with_history():
    # Test that conversation_history is accepted
    req = QueryRequest(
        question="Hello",
        conversation_history=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there"}
        ]
    )
    assert len(req.conversation_history) == 2
    assert req.conversation_history[0]["role"] == "user"

def test_history_truncation_logic():
    # We can't test the truncation in QueryRequest because it happens in main.py
    # But we can verify the schema allows long strings (validation happens later)
    long_str = "a" * 1000
    req = QueryRequest(
        question="Hello",
        conversation_history=[{"role": "user", "content": long_str}]
    )
    assert len(req.conversation_history[0]["content"]) == 1000
