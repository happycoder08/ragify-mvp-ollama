"""
Grounding gate logic for RAGify.

Deterministic pre-LLM validation to ensure answers are grounded in retrieved evidence.
Prevents hallucinations by refusing queries that lack sufficient supporting evidence.
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Common English stopwords for filtering (grounding-specific)
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
    'what', 'when', 'where', 'who', 'which', 'why', 'how', 'do', 'does', 'did',
    'have', 'had', 'should', 'could', 'would', 'can', 'may', 'i', 'my', 'me'
}

# Grounding gate constants
MIN_SUPPORT = 2  # Minimum single-line overlap count for evidence to proceed to LLM
MIN_TOTAL_SUPPORT = 4  # Minimum sum of top 3 overlaps across all evidence lines
MAX_EVIDENCE_LINES_TOTAL = 6  # Maximum total evidence lines across all chunks
MAX_EVIDENCE_LINES_PER_CHUNK = 3  # Maximum evidence lines to extract per chunk


def _tokenize_and_filter(text: str, min_len: int = 2) -> list:
    """
    Tokenize text and remove stopwords.
    Returns list (not set) to preserve term frequency for BM25-style scoring.
    
    Args:
        text: Text to tokenize
        min_len: Minimum token length to keep
    
    Returns:
        List of tokens with stopwords removed
    """
    cleaned = ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in text)
    tokens = [t for t in cleaned.split() if len(t) > min_len and t not in STOPWORDS]
    return tokens


def extract_evidence_lines(chunk_text: str, question: str, max_lines: int = MAX_EVIDENCE_LINES_PER_CHUNK) -> list[tuple[str, int]]:
    """
    Extract top evidence lines from a chunk based on lexical overlap with question.
    Returns list of (line, overlap_count) tuples sorted by relevance, up to max_lines.
    
    Token-based filtering: excludes lines with <2 tokens unless they contain digits/time markers.
    Tie-breaking: prefers bullets, anchor patterns (digits/times), and lines near headers.
    
    Args:
        chunk_text: The text of the chunk to extract lines from
        question: The user's question (for computing overlap)
        max_lines: Maximum number of lines to return (default: MAX_EVIDENCE_LINES_PER_CHUNK)
    
    Returns:
        List of (line, overlap_count) tuples, sorted by overlap score descending
    """
    if not chunk_text or not question:
        return []
    
    # Normalize question to tokens (lowercase, remove stopwords)
    q_tokens = set(_tokenize_and_filter(question))
    if not q_tokens:
        return []
    
    # Split chunk into lines and score each
    lines = chunk_text.split('\n')
    scored_lines = []
    
    for idx, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Token-based filter: exclude lines with <2 tokens unless they have digits/time markers
        line_tokens = _tokenize_and_filter(line_stripped)
        has_anchor = bool(re.search(r'\b\d+', line_stripped) or re.search(r'\b\d{1,2}:\d{2}', line_stripped))
        
        if len(line_tokens) < 2 and not has_anchor:
            continue  # Skip lines with insufficient content
        
        # Tokenize line for overlap
        line_token_set = set(line_tokens)
        if not line_token_set:
            continue
        
        # Compute overlap count
        overlap = len(q_tokens & line_token_set)
        if overlap > 0:
            # Compute tie-breaker score components
            is_bullet = bool(re.match(r'^\s*(?:[-*•]|\d+[.)])', line))
            is_near_header = (idx == 0 or (idx > 0 and lines[idx-1].strip().endswith(':')))
            
            # Tie-breaker tuple: (overlap, is_bullet, has_anchor, is_near_header, line_length)
            # Higher values sort first
            tie_breaker = (
                overlap,
                1 if is_bullet else 0,
                1 if has_anchor else 0,
                1 if is_near_header else 0,
                len(line_stripped)
            )
            
            scored_lines.append((line_stripped, overlap, tie_breaker))
    
    # Sort by tie_breaker tuple (descending)
    scored_lines.sort(key=lambda x: x[2], reverse=True)
    
    # Return top max_lines as (line, overlap) tuples
    return [(line, overlap) for line, overlap, _ in scored_lines[:max_lines]]


def _compute_grounding_gate(
    question: str,
    selected_chunks: list[tuple[str, dict, float]],
    chunk_ids: list[str]
) -> tuple[bool, str, list[str], float, float, str]:
    """
    Deterministic grounding gate: check if retrieved chunks have sufficient evidence.
    
    Uses global aggregation: extracts evidence from all chunks, then selects top MAX_EVIDENCE_LINES_TOTAL
    lines globally with stable ordering.
    
    Gate conditions (ALL must pass):
    1. Evidence lines non-empty
    2. Max overlap >= MIN_SUPPORT (at least one line has strong match)
    3. Sum of top 3 overlaps >= MIN_TOTAL_SUPPORT (cumulative evidence strength)
    4. Numeric/time questions need numeric/time anchors in evidence
    
    Args:
        question: User's question
        selected_chunks: List of (doc_text, metadata, distance) tuples
        chunk_ids: List of chunk IDs corresponding to selected_chunks
    
    Returns:
        Tuple of (should_proceed, refusal_reason, evidence_lines, max_overlap, sum_top3, failed_check)
        - should_proceed: True if evidence is sufficient, False to refuse
        - refusal_reason: "NOT_FOUND" if refused, empty string otherwise
        - evidence_lines: List of extracted evidence lines (text only)
        - max_overlap: Max overlap count across all evidence lines
        - sum_top3: Sum of top 3 overlap counts
        - failed_check: "NO_EVIDENCE", "LOW_SUPPORT", "MISSING_ANCHOR", or "" if passed
    """
    # Extract evidence lines from all selected chunks (with overlap scores)
    all_evidence_tuples = []
    for doc, meta, dist in selected_chunks:
        chunk_tuples = extract_evidence_lines(doc, question, max_lines=MAX_EVIDENCE_LINES_PER_CHUNK)
        all_evidence_tuples.extend(chunk_tuples)
    
    # Check 1: No evidence lines extracted
    if not all_evidence_tuples:
        return False, "NOT_FOUND", [], 0.0, 0.0, "NO_EVIDENCE"
    
    # Global selection: sort all evidence tuples by overlap (descending) and take top MAX_EVIDENCE_LINES_TOTAL
    # Sorting is already stable from extract_evidence_lines (includes tie-breakers)
    all_evidence_tuples.sort(key=lambda x: x[1], reverse=True)
    top_evidence_tuples = all_evidence_tuples[:MAX_EVIDENCE_LINES_TOTAL]
    
    # Extract overlap scores for threshold checks
    overlap_scores = [overlap for _, overlap in top_evidence_tuples]
    max_overlap = max(overlap_scores) if overlap_scores else 0
    sum_top3 = sum(sorted(overlap_scores, reverse=True)[:3])
    
    # Extract text lines for return value
    evidence_lines = [line for line, _ in top_evidence_tuples]
    
    # Check 2: Max overlap below minimum
    if max_overlap < MIN_SUPPORT:
        return False, "NOT_FOUND", evidence_lines, max_overlap, sum_top3, "LOW_SUPPORT"
    
    # Check 3: Sum of top 3 overlaps below minimum total support
    if sum_top3 < MIN_TOTAL_SUPPORT:
        return False, "NOT_FOUND", evidence_lines, max_overlap, sum_top3, "LOW_SUPPORT"
    
    # Check 4: Numeric/time-sensitive questions need numeric/time anchors
    q_lower = question.lower()
    # Detect numeric/time questions
    has_numeric_question = bool(
        re.search(r'\b\d+', question) or  # Contains digits
        any(word in q_lower for word in ['time', 'when', 'hour', 'day', 'days', 'week', 'month', 'year', 'am', 'pm'])
    )
    
    if has_numeric_question:
        # Check if evidence contains numeric/time patterns
        evidence_text = ' '.join(evidence_lines).lower()
        has_numeric_anchor = bool(
            re.search(r'\b\d+', evidence_text) or  # Contains digits
            re.search(r'\b\d{1,2}:\d{2}', evidence_text) or  # Time pattern
            re.search(r'\b(?:am|pm)\b', evidence_text)  # AM/PM
        )
        
        if not has_numeric_anchor:
            return False, "NOT_FOUND", evidence_lines, max_overlap, sum_top3, "MISSING_ANCHOR"
    
    # All checks passed
    return True, "", evidence_lines, max_overlap, sum_top3, ""
