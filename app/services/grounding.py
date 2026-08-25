"""
Grounding gate logic for RAGify.

Deterministic pre-LLM validation to ensure answers are grounded in retrieved evidence.
Prevents hallucinations by refusing queries that lack sufficient supporting evidence.
"""

import re
import logging
import os
from typing import List, Tuple, Optional

from app.config import settings

logger = logging.getLogger(__name__)

# Common English stopwords for filtering (grounding-specific)
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
    'what', 'when', 'where', 'who', 'which', 'why', 'how', 'do', 'does', 'did',
    'have', 'had', 'should', 'could', 'would', 'can', 'may', 'i', 'my', 'me'
}

# Deterministic numeric grounding helpers.
def extract_numeric_consensus(evidence_chunks: List[str]) -> Tuple[Optional[float], bool, List[float]]:
    values: List[float] = []
    for chunk in evidence_chunks or []:
        for match in re.findall(r"\b\d+(?:\.\d+)?\b", chunk or ""):
            try:
                val = float(match)
            except ValueError:
                continue
            # Filter out likely years to avoid policy-year conflicts.
            if 2020 <= val <= 2030 and val.is_integer():
                continue
            values.append(val)

    unique_vals = sorted(set(values))
    if not unique_vals:
        return None, False, []
    if len(unique_vals) == 1:
        return unique_vals[0], False, unique_vals
    return None, True, unique_vals


def validate_numeric_alignment(llm_answer: str, expected_val: float) -> bool:
    if llm_answer is None:
        return False
    # Allow integer and float formatting to match.
    normalized = llm_answer
    if expected_val.is_integer():
        target = str(int(expected_val))
        if target in normalized:
            return True
    target = str(expected_val)
    return target in normalized


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
    tokens = [t for t in cleaned.split() if len(t) > min_len and t.lower() not in STOPWORDS]
    return tokens


def extract_evidence_lines(
    chunk_text: str,
    question: str,
    max_lines: int = MAX_EVIDENCE_LINES_PER_CHUNK,
    target_field: str | None = None,
) -> list[tuple[str, int]]:
    """
    Extract top evidence lines from a chunk based on lexical overlap with question.
    Returns list of (line, overlap_count) tuples sorted by relevance, up to max_lines.
    
    Token-based filtering: excludes lines with <2 tokens unless they contain digits/time markers.
    Tie-breaking: prefers bullets, anchor patterns (digits/times), and lines near headers.
    Numeric/time bonus: +1 to overlap if question is time-sensitive AND line contains time/numeric anchor.
    
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
    
    # Detect if question is time/numeric sensitive
    q_lower = question.lower()
    is_time_sensitive = bool(
        re.search(r'\b\d+', question) or  # Contains digits
        any(word in q_lower for word in ['time', 'when', 'hour', 'day', 'days', 'week', 'month', 'year', 'am', 'pm', 'arrive', 'arrival'])
    )
    
    # Split chunk into lines and score each
    lines = chunk_text.split('\n')
    scored_lines = []
    target_keywords = []
    if target_field:
        target_map = {
            "VACATION": ["vacation", "pto", "paid time off", "time off"],
            "SICK": ["sick", "sick time", "sick leave", "illness"],
        }
        target_keywords = target_map.get(target_field, [])

    def _is_section_header(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if stripped.endswith(":"):
            return True
        return stripped.isupper() and len(stripped) <= 60

    def _mentions_target(line: str) -> bool:
        lower_line = line.lower()
        return any(kw in lower_line for kw in target_keywords)

    in_target_section = False
    
    for idx, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        if target_keywords and _is_section_header(line_stripped):
            in_target_section = _mentions_target(line_stripped)
            continue

        # Token-based filter: exclude lines with <2 tokens unless they have digits/time markers
        line_tokens = _tokenize_and_filter(line_stripped)
        has_anchor = bool(
            re.search(r'\b\d+', line_stripped) or  # Contains digits
            re.search(r'\b\d{1,2}:\d{2}', line_stripped) or  # Time pattern HH:MM
            re.search(r'\b(?:am|pm)\b', line_stripped.lower())  # AM/PM
        )

        if len(line_tokens) < 2 and not has_anchor:
            if settings.TOKEN_OVERLAP_THRESHOLD <= 1 and line_token_set & q_tokens:
                pass
            else:
                continue  # Skip lines with insufficient content

        # Tokenize line for overlap
        line_token_set = set(line_tokens)
        raw_overlap = len(q_tokens & line_token_set)
        effective_overlap = raw_overlap
        anchor_bonus_applied = False

        # If time/numeric question and line has anchor, treat as explicit support (even if raw_overlap==0)
        if is_time_sensitive and has_anchor:
            if raw_overlap == 0:
                # Guardrail: treat as explicit support
                effective_overlap = 2  # At least MIN_SUPPORT
                anchor_bonus_applied = True
            else:
                effective_overlap = raw_overlap + 1
                anchor_bonus_applied = True
        if target_keywords and in_target_section:
            effective_overlap += 2
        if target_keywords and _mentions_target(line_stripped):
            effective_overlap += 1

        # Log raw vs effective overlap for debug
        logger.debug(
            "[GroundingGate] Line: '%s' | raw_overlap=%d | effective_overlap=%d | anchor_bonus=%s",
            line_stripped[:80], raw_overlap, effective_overlap, anchor_bonus_applied
        )

        if effective_overlap > 0:
            is_bullet = bool(re.match(r'^\s*(?:[-*•]|\d+[.)])', line))
            is_near_header = (idx == 0 or (idx > 0 and lines[idx-1].strip().endswith(':')))
            tie_breaker = (
                effective_overlap,
                1 if is_bullet else 0,
                1 if has_anchor else 0,
                1 if is_near_header else 0,
                len(line_stripped)
            )
            scored_lines.append((line_stripped, effective_overlap, tie_breaker))
    
    # Sort by tie_breaker tuple (descending)
    scored_lines.sort(key=lambda x: x[2], reverse=True)
    
    # Return top max_lines as (line, overlap) tuples
    return [(line, overlap) for line, overlap, _ in scored_lines[:max_lines]]


def _compute_grounding_gate(
    question: str,
    selected_chunks: list[tuple[str, dict, float]],
    chunk_ids: list[str],
    target_field: str | None = None,
) -> tuple[bool, str, list[str], float, float, str]:
    # --- Anchor-first fast-pass for time/numeric questions ---
    def _has_time_anchor(text: str) -> bool:
        t = text.lower()
        return bool(
            re.search(r'\b\d{1,2}:\d{2}\b', t) or          # 8:00
            re.search(r'\b\d{1,2}\s?(am|pm)\b', t) or      # 8am, 8 am
            re.search(r'\b\d{1,2}:\d{2}\s?(am|pm)\b', t)   # 8:00am, 8:00 am
        )

    def _question_is_time_like(q: str) -> bool:
        ql = q.lower()
        return any(w in ql for w in ["time", "when", "hour", "arrive", "arrival", "start"]) or bool(re.search(r'\b\d', ql))

    def _weak_intent_match(q: str, evidence: str) -> bool:
        ql = q.lower()
        el = evidence.lower()
        intent_terms = ["arrive", "arrival", "report", "start", "check-in", "begin", "office"]
        return any(t in ql and t in el for t in intent_terms)

    # (anchor-first fast-pass logic moved below after evidence_lines is defined)
        # Extract text lines for return value
        evidence_lines = [line for line, _ in top_evidence_tuples]

        # Anchor-first fast-pass: if question is time-like and any evidence line contains a time anchor (and weak intent match), pass immediately
        if _question_is_time_like(question):
            for line in evidence_lines:
                if _has_time_anchor(line) and _weak_intent_match(question, line):
                    return True, "", evidence_lines, max_overlap, sum_top3, ""
            # If you want it even more permissive (demo-friendly), drop _weak_intent_match:
            # if any(_has_time_anchor(line) for line in evidence_lines):
            #     return True, "", evidence_lines, max_overlap, sum_top3, ""
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
        - max_overlap: Max overlap count across all evidence lines (includes time/numeric bonus)
        - sum_top3: Sum of top 3 overlap counts (includes time/numeric bonus)
        - failed_check: "NO_EVIDENCE", "LOW_SUPPORT", "MISSING_ANCHOR", or "" if passed
    """
    # Detect if question is time/numeric sensitive (for logging)
    q_lower = question.lower()
    is_time_sensitive = bool(
        re.search(r'\b\d+', question) or
        any(word in q_lower for word in ['time', 'when', 'hour', 'day', 'days', 'week', 'month', 'year', 'am', 'pm', 'arrive', 'arrival'])
    )
    
    # Extract evidence lines from all selected chunks (with overlap scores)
    all_evidence_tuples = []
    for doc, meta, dist in selected_chunks:
        chunk_tuples = extract_evidence_lines(
            doc,
            question,
            max_lines=MAX_EVIDENCE_LINES_PER_CHUNK,
            target_field=target_field,
        )
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
    
    # Calculate Grounding Score (Ratio of overlap to unique query tokens)
    q_tokens = set(_tokenize_and_filter(question))
    q_len = len(q_tokens)
    grounding_score = (max_overlap / q_len) if q_len > 0 else 0.0

    # Extract text lines for return value
    evidence_lines = [line for line, _ in top_evidence_tuples]
    
    # Debug logging for grounding gate metrics
    logger.debug(
        "Grounding gate: time_sensitive=%s, max_overlap=%.0f (threshold=%d), score=%.2f (threshold=%.2f), top_scores=%s",
        is_time_sensitive, max_overlap, settings.TOKEN_OVERLAP_THRESHOLD, grounding_score, settings.GROUNDING_THRESHOLD,
        [round(s, 1) for s in overlap_scores[:3]]
    )
    
    # Check 2: Max overlap below minimum (Absolute Threshold)
    # Allow explicit support for TOKEN_OVERLAP_THRESHOLD if any evidence line has anchor and question is time/numeric
    explicit_support = False
    if is_time_sensitive:
        for line, overlap in top_evidence_tuples:
            has_anchor = bool(
                re.search(r'\b\d{1,2}:\d{2}\b', line.lower()) or re.search(r'\b\d{1,2}\s?(am|pm)\b', line.lower())
            )
            if has_anchor and overlap >= 1:
                explicit_support = True
                break
    
    # Apply Thresholds
    if max_overlap < settings.TOKEN_OVERLAP_THRESHOLD and not explicit_support:
        return False, "NOT_FOUND", evidence_lines, max_overlap, sum_top3, "LOW_SUPPORT"

    if grounding_score < settings.GROUNDING_THRESHOLD and not explicit_support:
        return False, "NOT_FOUND", evidence_lines, max_overlap, sum_top3, "LOW_SCORE"

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
