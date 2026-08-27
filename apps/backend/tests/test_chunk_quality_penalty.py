import pytest
from app.services.rag_service import _score_chunk_quality

def test_score_chunk_quality_normal():
    text = "This is a normal chunk of text that is long enough to not be penalized. It has punctuation and looks like a real sentence."
    penalty, reasons = _score_chunk_quality(text)
    assert penalty == 0.0
    assert reasons == []

def test_score_chunk_quality_short():
    text = "Too short."
    penalty, reasons = _score_chunk_quality(text)
    assert penalty < 0
    assert "quality:short_length" in reasons

def test_score_chunk_quality_header_like():
    text = "INTRODUCTION TO POLICY"
    penalty, reasons = _score_chunk_quality(text)
    assert penalty < 0
    assert "quality:header_like" in reasons

def test_score_chunk_quality_empty():
    text = "   "
    penalty, reasons = _score_chunk_quality(text)
    assert penalty == -10.0
    assert "quality:empty" in reasons
