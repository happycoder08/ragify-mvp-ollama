"""
Unit tests for the grounding gate feature.

Tests the deterministic evidence validation before LLM calls.
"""

import pytest
from app.services.rag_service import (
    extract_evidence_lines,
    _compute_grounding_gate,
    MIN_SUPPORT,
    MAX_EVIDENCE_LINES
)


class TestExtractEvidenceLines:
    """Test the line-level evidence extraction."""
    
    def test_empty_chunk(self):
        """Empty chunk should return empty list."""
        result = extract_evidence_lines("", "what is the policy?")
        assert result == []
    
    def test_short_lines_filtered(self):
        """Lines shorter than 10 chars should be filtered."""
        chunk = "Title\nShort\nThis is a longer line with policy information"
        result = extract_evidence_lines(chunk, "policy")
        assert len(result) == 1
        assert "policy information" in result[0]
    
    def test_lexical_overlap_scoring(self):
        """Lines with more query token overlap should rank higher."""
        chunk = """
The vacation policy allows 15 days per year.
Sick leave is handled separately.
For vacation requests, submit at least 2 weeks notice.
        """
        result = extract_evidence_lines(chunk, "vacation policy requests")
        # Should prioritize lines with multiple query tokens
        assert len(result) > 0
        assert "vacation" in result[0].lower()
    
    def test_max_lines_limit(self):
        """Should respect max_lines parameter."""
        chunk = "\n".join([f"Line {i} with some content here" for i in range(20)])
        result = extract_evidence_lines(chunk, "content", max_lines=3)
        assert len(result) <= 3
    
    def test_deterministic_sorting(self):
        """Same input should always produce same output (stable sorting)."""
        chunk = """
First line with important information here.
Second line also has important details.
Third line contains important facts too.
        """
        question = "important information"
        
        result1 = extract_evidence_lines(chunk, question)
        result2 = extract_evidence_lines(chunk, question)
        
        assert result1 == result2  # Deterministic


class TestComputeGroundingGate:
    """Test the grounding gate decision logic."""
    
    def test_empty_chunks_refused(self):
        """Empty chunks should be refused."""
        selected_chunks = []
        chunk_ids = []
        
        should_proceed, reason, lines, score = _compute_grounding_gate(
            "what is the policy?", selected_chunks, chunk_ids
        )
        
        assert should_proceed is False
        assert reason == "NOT_FOUND"
        assert lines == []
        assert score == 0.0
    
    def test_header_only_chunks_refused(self):
        """Chunks with only short headers (no content) should be refused."""
        selected_chunks = [
            ("POLICY\nHEADER", {"chunk": 0}, 0.5),
            ("GUIDE", {"chunk": 1}, 0.6)
        ]
        chunk_ids = ["chunk_0", "chunk_1"]
        
        should_proceed, reason, lines, score = _compute_grounding_gate(
            "what is the vacation policy?", selected_chunks, chunk_ids
        )
        
        # These short lines get filtered out (< 10 chars), no evidence extracted
        assert should_proceed is False
        assert reason == "NOT_FOUND"
    
    def test_low_support_score_refused(self):
        """Chunks with support_score < MIN_SUPPORT should be refused."""
        # Create chunk with only 1 token overlap (below MIN_SUPPORT=2)
        selected_chunks = [
            ("This chunk talks about something unrelated to the query topic.", {"chunk": 0}, 0.5)
        ]
        chunk_ids = ["chunk_0"]
        
        should_proceed, reason, lines, score = _compute_grounding_gate(
            "vacation policy details", selected_chunks, chunk_ids
        )
        
        # Should have some evidence lines but low support
        assert should_proceed is False
        assert reason == "NOT_FOUND"
        assert score < MIN_SUPPORT
    
    def test_strong_evidence_passes(self):
        """Chunks with good lexical overlap should pass."""
        selected_chunks = [
            (
                "The vacation policy allows employees to take 15 days per year. "
                "Vacation requests must be submitted at least 2 weeks in advance.",
                {"chunk": 0},
                0.3
            )
        ]
        chunk_ids = ["chunk_0"]
        
        should_proceed, reason, lines, score = _compute_grounding_gate(
            "vacation policy", selected_chunks, chunk_ids
        )
        
        assert should_proceed is True
        assert reason == ""
        assert len(lines) > 0
        assert score >= MIN_SUPPORT
    
    def test_numeric_question_without_numeric_evidence_refused(self):
        """Numeric questions need numeric anchors in evidence."""
        selected_chunks = [
            (
                "The vacation policy is generous and allows employees to take time off. "
                "Submit your request to your manager for approval.",
                {"chunk": 0},
                0.3
            )
        ]
        chunk_ids = ["chunk_0"]
        
        # Question asks for numeric info (15 days) but chunk has no numbers
        should_proceed, reason, lines, score = _compute_grounding_gate(
            "how many vacation days do I get?", selected_chunks, chunk_ids
        )
        
        # Should refuse due to missing numeric anchor
        assert should_proceed is False
        assert reason == "NOT_FOUND"
    
    def test_numeric_question_with_numeric_evidence_passes(self):
        """Numeric questions with numeric anchors should pass."""
        selected_chunks = [
            (
                "The vacation policy allows employees to take 15 days per year. "
                "Sick leave provides an additional 10 days annually.",
                {"chunk": 0},
                0.3
            )
        ]
        chunk_ids = ["chunk_0"]
        
        should_proceed, reason, lines, score = _compute_grounding_gate(
            "how many vacation days", selected_chunks, chunk_ids
        )
        
        assert should_proceed is True
        assert reason == ""
        assert score >= MIN_SUPPORT
    
    def test_time_question_without_time_evidence_refused(self):
        """Time-sensitive questions need time patterns in evidence."""
        selected_chunks = [
            (
                "The onboarding process covers many important topics. "
                "You will meet with your manager and HR team.",
                {"chunk": 0},
                0.3
            )
        ]
        chunk_ids = ["chunk_0"]
        
        should_proceed, reason, lines, score = _compute_grounding_gate(
            "when does onboarding start?", selected_chunks, chunk_ids
        )
        
        # Should refuse: 'when' question but no time info
        assert should_proceed is False
        assert reason == "NOT_FOUND"
    
    def test_time_question_with_time_evidence_passes(self):
        """Time questions with time patterns should pass."""
        selected_chunks = [
            (
                "Onboarding starts at 9:00 AM on your first day. "
                "Plan to arrive 15 minutes early for check-in.",
                {"chunk": 0},
                0.3
            )
        ]
        chunk_ids = ["chunk_0"]
        
        should_proceed, reason, lines, score = _compute_grounding_gate(
            "when does onboarding start?", selected_chunks, chunk_ids
        )
        
        assert should_proceed is True
        assert reason == ""
    
    def test_multiple_chunks_aggregation(self):
        """Should aggregate evidence from multiple chunks."""
        selected_chunks = [
            ("The vacation policy is comprehensive.", {"chunk": 0}, 0.3),
            ("Employees receive 15 days of vacation per year.", {"chunk": 1}, 0.4),
            ("Submit requests at least 2 weeks in advance.", {"chunk": 2}, 0.5)
        ]
        chunk_ids = ["chunk_0", "chunk_1", "chunk_2"]
        
        should_proceed, reason, lines, score = _compute_grounding_gate(
            "vacation policy", selected_chunks, chunk_ids
        )
        
        # Should combine evidence from all chunks
        assert should_proceed is True
        assert len(lines) > 0  # Multiple lines extracted
        assert score >= MIN_SUPPORT
    
    def test_deterministic_behavior(self):
        """Same input should always produce same output."""
        selected_chunks = [
            (
                "The vacation policy allows 15 days per year. "
                "Submit requests 2 weeks in advance.",
                {"chunk": 0},
                0.3
            )
        ]
        chunk_ids = ["chunk_0"]
        question = "vacation policy"
        
        result1 = _compute_grounding_gate(question, selected_chunks, chunk_ids)
        result2 = _compute_grounding_gate(question, selected_chunks, chunk_ids)
        
        assert result1 == result2  # Fully deterministic


class TestGroundingGateEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_exactly_min_support(self):
        """Support score exactly at MIN_SUPPORT should pass."""
        # Craft chunk to have exactly MIN_SUPPORT tokens overlap
        selected_chunks = [
            (
                f"The vacation policy information is available in the handbook here.",
                {"chunk": 0},
                0.3
            )
        ]
        chunk_ids = ["chunk_0"]
        
        should_proceed, reason, lines, score = _compute_grounding_gate(
            "vacation policy", selected_chunks, chunk_ids
        )
        
        # With "vacation" and "policy" both present, score should be >= 2
        assert score >= MIN_SUPPORT
        assert should_proceed is True
    
    def test_stopwords_filtered(self):
        """Common stopwords should not contribute to overlap."""
        selected_chunks = [
            (
                "This is a document about the important policy that you need to know.",
                {"chunk": 0},
                0.3
            )
        ]
        chunk_ids = ["chunk_0"]
        
        # Query with many stopwords
        should_proceed, reason, lines, score = _compute_grounding_gate(
            "what is the policy that I need", selected_chunks, chunk_ids
        )
        
        # Overlap should be based on content words (policy, need) not stopwords
        assert len(lines) > 0
    
    def test_case_insensitive(self):
        """Matching should be case-insensitive."""
        selected_chunks = [
            ("THE VACATION POLICY ALLOWS 15 DAYS.", {"chunk": 0}, 0.3)
        ]
        chunk_ids = ["chunk_0"]
        
        should_proceed, reason, lines, score = _compute_grounding_gate(
            "vacation policy", selected_chunks, chunk_ids
        )
        
        assert should_proceed is True
        assert score >= MIN_SUPPORT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
