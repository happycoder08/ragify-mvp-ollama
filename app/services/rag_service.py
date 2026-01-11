from dataclasses import dataclass
import json
from typing import Any, Dict, List, Tuple, AsyncGenerator, Optional

from dataclasses import dataclass, field
from app.schemas.query import AnswerSchema, AnswerDecision, DecisionType

def get_collection_sync(tenant_id: str = "default"):
    import asyncio
    return asyncio.get_event_loop().run_until_complete(get_collection_async(tenant_id))

@dataclass
class ChunkHit:
    chunk_id: str
    doc: str
    meta: Dict[str, Any]
    dist: float
    lexical_score: float = field(default=0.0)
    final_score: float = field(default=0.0)
    embedding: List[float] = field(default_factory=list)
    why_selected: List[str] = field(default_factory=list)

# --- ChunkHit utilities: header key, rerank, dedupe ---
def _apply_mmr_selection(hits: List[ChunkHit], question_emb: List[float], k: int, lambda_param: float = 0.6) -> List[ChunkHit]:
    """
    Apply Maximal Marginal Relevance (MMR) to select diverse chunks.
    MMR score = lambda * sim(query, chunk) - (1-lambda) * max_sim(chunk, selected_chunks)
    """
    if not hits or k <= 0:
        return []
    
    # Start with the most relevant chunk (highest similarity to query)
    selected = [hits[0]]  # hits are already sorted by relevance
    remaining = hits[1:]
    
    while len(selected) < k and remaining:
        best_score = -float('inf')
        best_idx = -1
        
        for i, candidate in enumerate(remaining):
            # Similarity to query (lower dist = higher similarity, so use 1 - dist as proxy)
            query_sim = 1.0 - candidate.dist
            
            # Maximum similarity to already selected chunks
            max_sim_to_selected = 0.0
            if candidate.embedding and all(h.embedding for h in selected):
                for sel in selected:
                    # Cosine similarity between candidate and selected chunk
                    sim = _cosine_similarity(candidate.embedding, sel.embedding)
                    max_sim_to_selected = max(max_sim_to_selected, sim)
            
            # MMR score
            mmr_score = lambda_param * query_sim - (1 - lambda_param) * max_sim_to_selected
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        
        if best_idx >= 0:
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
        else:
            break
    
    return selected

def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0

def _apply_header_reranking(hits: List[ChunkHit], question: str, debug: bool = False, request_id: str = None) -> List[ChunkHit]:
    q = question.lower()
    def score(hit: ChunkHit) -> tuple:
        hk = str(hit.meta.get("header") or hit.meta.get("section") or "").strip().lower()
        header_match = hk in q if hk else False
        return (0 if header_match else 1, hit.dist)
    return sorted(hits, key=score)

def _dedupe_results(hits: List[ChunkHit], ids: List[str] = None) -> List[ChunkHit]:
    seen = set()
    out: List[ChunkHit] = []
    for h in hits:
        if h.chunk_id not in seen:
            seen.add(h.chunk_id)
            out.append(h)
    return out

def _dedupe_by_header(hits: List[ChunkHit], ids: List[str] = None, max_per_header: int = 1) -> List[ChunkHit]:
    buckets: Dict[str, List[ChunkHit]] = {}
    for h in hits:
        hk = str(h.meta.get("header") or h.meta.get("section") or "").strip().lower()
        if hk not in buckets:
            buckets[hk] = []
        if len(buckets[hk]) < max_per_header:
            buckets[hk].append(h)
    order = []
    seen = set()
    for h in hits:
        hk = str(h.meta.get("header") or h.meta.get("section") or "").strip().lower()
        if hk not in seen:
            seen.add(hk)
            order.append(hk)
    out: List[ChunkHit] = []
    for hk in order:
        out.extend(buckets[hk])
    return out

def _get_anchor_type(doc: str) -> str | None:
    """Determine anchor type for a document chunk."""
    doc_lower = doc.lower()
    
    # Check for WiFi anchors
    if any(kw in doc_lower for kw in ["ssid", "wifi", "password"]):
        return "WIFI"
    
    # Check for time anchors using the same regex as elsewhere
    import re
    time_regex = r"\b\d{1,2}(:\d{2})?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])|\b\d{1,2}\s*(a\.?m\.?|p\.?m\.?)(?![a-z])|\b\d{1,2}:\d{2}\b"
    if re.search(time_regex, doc_lower, re.IGNORECASE):
        return "TIME"
    
    # Check for manager/supervisor mentions
    if any(kw in doc_lower for kw in ["manager", "supervisor", "boss"]):
        return "MANAGER"
    
    # Check for badge/security mentions
    if any(kw in doc_lower for kw in ["badge", "id card", "security badge"]):
        return "BADGE"
    
    # Check for reception/front desk mentions
    if any(kw in doc_lower for kw in ["reception", "front desk", "lobby"]):
        return "RECEPTION"
    
    return None

def _is_broad_question(question: str) -> bool:
    """Detect if a question is broad and would benefit from more context."""
    q_lower = question.lower()
    broad_indicators = [
        "what do i do", "first day", "overview", "walk me through", 
        "steps", "checklist", "when does", "when do", "how do",
        "what time", "what is", "tell me about"
    ]
    return any(indicator in q_lower for indicator in broad_indicators)

def _get_debug_anchor_type(doc: str) -> str | None:
    """Determine anchor type for debug objects (retrieved_chunks_top20 and selected_chunks)."""
    doc_lower = doc.lower()
    import re
    
    # Check for WiFi anchors
    if any(kw in doc_lower for kw in ["wifi", "ssid", "password"]):
        return "WIFI"
    
    # Check for time anchors using the same regex as elsewhere
    time_regex = r"\b\d{1,2}(:\d{2})?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])|\b\d{1,2}\s*(a\.?m\.?|p\.?m\.?)(?![a-z])|\b\d{1,2}:\d{2}\b"
    if re.search(time_regex, doc_lower, re.IGNORECASE):
        return "TIME"
    
    # Check for location anchors
    if any(kw in doc_lower for kw in ["floor", "reception", "address"]):
        return "LOCATION"
    
    return None

def _extract_header_first_line(question: str, doc: str) -> str:
    """Extract the most relevant first line from a chunk based on the question context."""
    # If question contains "wifi", scan for lines containing wifi-related keywords
    if "wifi" in question.lower():
        for line in doc.splitlines():
            line_stripped = line.strip()
            if line_stripped and any(kw in line_stripped.lower() for kw in ["wifi", "ssid", "password"]):
                return line_stripped
    
    # Default behavior: return first non-empty line
    return next((ln.strip() for ln in doc.splitlines() if ln.strip()), "")

def _hits_from_chroma(res: Dict[str, Any]) -> List[ChunkHit]:
    docs = res.get("documents")
    metas = res.get("metadatas")
    dists = res.get("distances")
    ids = res.get("ids")
    embeddings = res.get("embeddings")
    if isinstance(docs, list) and docs and isinstance(docs[0], list):
        docs = docs[0]
    if isinstance(metas, list) and metas and isinstance(metas[0], list):
        metas = metas[0]
    if isinstance(dists, list) and dists and isinstance(dists[0], list):
        dists = dists[0]
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
        embeddings = embeddings[0]

    # Ensure all lists are iterable and have matching lengths.
    docs_list = docs or []
    metas_list = metas or []
    dists_list = dists or []
    ids_list = ids or []
    emb_list = embeddings or []

    n = max(len(docs_list), len(metas_list), len(dists_list), len(ids_list), len(emb_list))
    # Pad shorter lists with sensible defaults
    def _pad(lst, default):
        if not lst:
            return [default] * n
        if len(lst) < n:
            return list(lst) + [default] * (n - len(lst))
        return list(lst)

    docs_p = _pad(docs_list, "")
    metas_p = _pad(metas_list, {})
    dists_p = _pad(dists_list, 1.0)
    ids_p = _pad(ids_list, "")
    emb_p = _pad(emb_list, [])

    hits: List[ChunkHit] = []
    for doc, meta, dist, cid, emb in zip(docs_p, metas_p, dists_p, ids_p, emb_p):
        if not doc or not cid:
            continue
        hits.append(ChunkHit(chunk_id=str(cid), doc=str(doc), meta=meta or {}, dist=float(dist), embedding=emb or []))
    return hits

import time
import os
import logging
import re
import httpx

from . import clients
from .llm_providers import create_llm_provider, LLMProvider
from .reranker_providers import create_reranker_provider, RerankerProvider
from .grounding import (
    extract_evidence_lines,
    extract_numeric_consensus,
    validate_numeric_alignment,
    _compute_grounding_gate,
    MIN_SUPPORT,
    MIN_TOTAL_SUPPORT,
    MAX_EVIDENCE_LINES_TOTAL,
    MAX_EVIDENCE_LINES_PER_CHUNK,
)
from .validation import answer_supported_by_evidence
from .llm_orchestrator import generate_answer_stream
from app.guardrails import get_guardrail_config
from app.config import (
    REQUEST_TIMEOUT,
    SIMILARITY_THRESHOLD,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_TOKENS_FAST,
    MAX_TOKENS_FULL,
    TOP_K_FAST,
    TOP_K_FULL,
    TOP_N_FAST,
    TOP_N_FULL,
    EMBEDDING_MODEL,
    ENABLE_TIMING_LOGS,
    ENABLE_RERANKING,
    RERANKER_TOP_N,
    CONTEXT_BUDGET_CHARS,
    RAGIFY_MODE,
)

OLLAMA_BASE_URL = "http://localhost:11434"

logger = logging.getLogger(__name__)

# Global LLM provider instance (initialized on first use)
_llm_provider: LLMProvider = None

# Global reranker provider instance (initialized on first use)
_reranker_provider: RerankerProvider = None

# Global embedding provider instance (initialized on first use)
_embedding_provider = None

# Common English stopwords for filtering
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
    'what', 'when', 'where', 'who', 'which', 'why', 'how', 'do', 'does', 'did',
    'have', 'had', 'should', 'could', 'would', 'can', 'may', 'i', 'my', 'me'
}

# Lightweight intent synonym expansion for operational queries
INTENT_SYNONYMS = {
    "arrive": ["arrival", "report", "checkin"],
    "arrival": ["arrive", "report", "checkin"],
    "report": ["arrive", "arrival", "checkin"],
    "reception": ["frontdesk", "front", "desk"],
    "frontdesk": ["reception"],
    "front": ["reception"],
    "desk": ["reception"],
    "firstday": ["day1"],
    "day1": ["firstday"],
}

BENEFITS_FIELD_KEYWORDS = {
    "VACATION": ["vacation", "pto", "paid time off", "time off"],
    "SICK": ["sick", "sick time", "sick leave", "illness"],
}

SLOT_UNIT_KEYWORDS = {
    "VACATION": ["vacation day", "vacation days", "vacation", "pto", "paid time off"],
    "SICK": ["sick day", "sick days", "sick time", "sick leave"],
}


def _format_unit_for_slot(value: float, slot_label: str) -> str:
    if not slot_label:
        return ""
    unit_candidates = SLOT_UNIT_KEYWORDS.get(slot_label, [])
    unit = None
    for kw in unit_candidates:
        if "day" in kw:
            unit = kw
            break
    if not unit and unit_candidates:
        unit = unit_candidates[0]
    if not unit:
        return ""
    if "day" in unit:
        prefix = unit.replace("days", "").replace("day", "").strip()
        suffix = "day" if abs(value - 1.0) < 1e-9 else "days"
        return f"{prefix} {suffix}".strip()
    return unit


def _detect_benefits_target_field(question: str) -> Optional[str]:
    if not question:
        return None
    q_lower = question.lower()
    for field, keywords in BENEFITS_FIELD_KEYWORDS.items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", q_lower):
                return field
    return None


def _fact_single_misaligned_to_target(answer_text: str, target_field: str) -> bool:
    if not answer_text or not target_field:
        return False
    answer_lower = answer_text.lower()
    target_keywords = BENEFITS_FIELD_KEYWORDS.get(target_field, [])
    target_present = any(kw in answer_lower for kw in target_keywords)
    if target_present:
        return False
    for field, keywords in BENEFITS_FIELD_KEYWORDS.items():
        if field == target_field:
            continue
        if any(kw in answer_lower for kw in keywords):
            return True
    return True


def _slice_text_to_slot(snippet: str, slot_label: str) -> tuple[str, bool]:
    if not snippet or not slot_label:
        return snippet or "", False
    slot_keywords = BENEFITS_FIELD_KEYWORDS.get(slot_label, [])
    if not slot_keywords:
        return snippet, False

    lines = [ln.strip() for ln in snippet.splitlines()]
    start_idx = None
    for idx, line in enumerate(lines):
        if not line:
            continue
        if line.endswith(":") or (line.isupper() and len(line) <= 60):
            lower_line = line.lower()
            if any(kw in lower_line for kw in slot_keywords):
                start_idx = idx + 1
                break

    if start_idx is None:
        return snippet, False

    sliced_lines = []
    for line in lines[start_idx:]:
        if not line:
            continue
        if line.endswith(":") or (line.isupper() and len(line) <= 60):
            break
        sliced_lines.append(line)

    if not sliced_lines:
        return snippet, False
    return "\n".join(sliced_lines), True


def _is_numeric_fact_question(question: str) -> bool:
    if not question:
        return False
    q_lower = question.lower()
    if re.search(r"\b\d+\b", q_lower):
        return True
    return any(term in q_lower for term in ["how many", "days", "per year", "per week", "per month"])


def _extract_slot_numeric_line(
    question: str,
    evidence_items: List[Any],
    slot_label: str,
    context_text: Optional[str] = None,
) -> tuple[Optional[str], bool]:
    if not evidence_items or not slot_label:
        return None, False

    slot_keywords = SLOT_UNIT_KEYWORDS.get(slot_label, [])
    if not slot_keywords:
        return None, False

    years = set(re.findall(r"\b20\d{2}\b", question or ""))
    if years:
        filtered = []
        for ev in evidence_items:
            snippet = getattr(ev, "snippet", "") or ""
            heading = getattr(ev, "heading", "") or ""
            if any(y in snippet or y in heading for y in years):
                filtered.append(ev)
        if filtered:
            evidence_items = filtered

    number_pattern = re.compile(r"\b\d+(?:\.\d+)?\b")
    slot_slice_applied = False

    def _matches_slot(line: str) -> bool:
        lower_line = line.lower()
        return any(kw in lower_line for kw in slot_keywords)

    def _extract_from_text(text: str) -> Optional[str]:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if _matches_slot(line) and number_pattern.search(line):
                return re.split(r"(?<=[.!?])\s+", line)[0].strip()
        return None

    for ev in evidence_items:
        snippet = getattr(ev, "snippet", "") or ""
        sliced, sliced_applied = _slice_text_to_slot(snippet, slot_label)
        slot_slice_applied = slot_slice_applied or sliced_applied
        found = _extract_from_text(sliced)
        if found:
            return found, slot_slice_applied

    if context_text:
        context_lines = []
        for line in context_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("[CHUNK_ID=") or stripped == "----":
                continue
            context_lines.append(stripped)
        combined = "\n".join(context_lines)
        sliced, sliced_applied = _slice_text_to_slot(combined, slot_label)
        slot_slice_applied = slot_slice_applied or sliced_applied
        found = _extract_from_text(sliced)
        if found:
            return found, slot_slice_applied

    return None, slot_slice_applied


def _extract_targeted_fact_single_from_evidence(
    question: str,
    evidence_items: List[Any],
    target_field: str,
    context_text: Optional[str] = None,
) -> Optional[str]:
    if not evidence_items or not target_field:
        return None

    years = set(re.findall(r"\b20\d{2}\b", question or ""))
    if years:
        filtered = []
        for ev in evidence_items:
            snippet = getattr(ev, "snippet", "") or ""
            heading = getattr(ev, "heading", "") or ""
            if any(y in snippet or y in heading for y in years):
                filtered.append(ev)
        if filtered:
            evidence_items = filtered

    number_pattern = re.compile(r"\b\d+(?:\.\d+)?\b")
    target_keywords = BENEFITS_FIELD_KEYWORDS.get(target_field, [])
    if not target_keywords:
        return None

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

    def _extract_from_lines(lines: List[str]) -> Optional[str]:
        in_target_section = False
        for line in lines:
            if not line:
                continue
            if _is_section_header(line):
                in_target_section = _mentions_target(line)
                continue
            if in_target_section and _mentions_target(line) and number_pattern.search(line):
                return re.split(r"(?<=[.!?])\s+", line)[0].strip()

        for line in lines:
            if not line:
                continue
            if _mentions_target(line) and number_pattern.search(line):
                return re.split(r"(?<=[.!?])\s+", line)[0].strip()
        return None

    for ev in evidence_items:
        snippet = getattr(ev, "snippet", "") or ""
        snippet, _ = _slice_text_to_slot(snippet, target_field)
        lines = [ln.strip() for ln in snippet.splitlines()]
        extracted = _extract_from_lines(lines)
        if extracted:
            return extracted

    if context_text:
        context_lines = []
        for line in context_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("[CHUNK_ID=") or stripped == "----":
                continue
            context_lines.append(stripped)
        combined = "\n".join(context_lines)
        combined, _ = _slice_text_to_slot(combined, target_field)
        extracted = _extract_from_lines(combined.splitlines())
        if extracted:
            return extracted

    return None

def _fingerprint_chunk(text: str) -> str:
    """
    Create a deterministic fingerprint for a chunk to deduplicate near-identical content.
    Steps: lowercase -> remove punctuation -> collapse whitespace -> trim to 200 chars -> sha1.
    """
    import re
    import hashlib

    # Normalize: lowercase first
    normalized = text.lower()

    # Canonicalize common time formats so "8:00 AM" and "8 am" hash the same
    def _normalize_time(match: re.Match) -> str:
        hour = int(match.group(1))
        minute_str = match.group(2)
        minute = int(minute_str) if minute_str else 0
        suffix = match.group(3)
        return f"{hour:02d}{minute:02d}{suffix}"

    normalized = re.sub(r"\b(\d{1,2})\s*[:\.]?\s*(\d{0,2})\s*(am|pm)\b", _normalize_time, normalized)

    # Strip punctuation, collapse whitespace
    normalized = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", normalized)).strip()
    normalized = normalized[:200]
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

def log_timing_rag(event: str, duration: float, tenant_id: str, **extra):
    """Log timing events with structured JSON (RAG service)."""
    if not ENABLE_TIMING_LOGS:
        return
    log_data = {
        "event": event,
        "duration_ms": round(duration * 1000, 2),
        "tenant_id": tenant_id,
        **extra
    }
    logger.info(json.dumps(log_data))


def _normalize_token(token: str) -> str:
    """Deterministically normalize a token (lowercase, strip punctuation, trim simple suffixes)."""
    tok = re.sub(r"[^a-z0-9]", "", token.lower())
    for suffix in ("ing", "ed", "s"):
        if len(tok) > 3 and tok.endswith(suffix):
            tok = tok[: -len(suffix)]
            break
    return tok


def _tokenize_and_filter(text: str, min_len: int = 2, expand_intents: bool = False) -> list:
    """
    Tokenize text, normalize, optionally expand intent synonyms, and remove stopwords.
    Returns list (not set) to preserve term frequency when needed.
    """
    raw_tokens = re.split(r"\s+", re.sub(r"[^A-Za-z0-9]+", " ", text))
    normalized = []
    for tok in raw_tokens:
        norm = _normalize_token(tok)
        if not norm:
            continue
        if len(norm) <= min_len:
            continue
        if norm in STOPWORDS:
            continue
        normalized.append(norm)

    if expand_intents:
        expanded: List[str] = []
        for tok in normalized:
            expanded.append(tok)
            expanded.extend(INTENT_SYNONYMS.get(tok, []))
        normalized = expanded

    return normalized


def _lexical_overlap_score(query: str, doc: str) -> float:
    """
    Compute lexical overlap score between query and document.
    Weigh header overlap higher than body overlap and expand operational intents.
    Returns a score between 0 and 1+ (boosted).
    """

    # Custom stopwords for generic tokens
    GENERIC_STOPWORDS = {"first", "day", "days"}

    # Tokenize with stopword removal and intent expansion
    query_tokens = [t for t in _tokenize_and_filter(query, expand_intents=True) if t not in GENERIC_STOPWORDS]
    if not query_tokens:
        return 0.0

    doc_lines = doc.split('\n')
    header_line = ""
    body_lines: List[str] = []
    for line in doc_lines:
        if line.strip() and not header_line:
            header_line = line
            continue
        if header_line:
            body_lines.append(line)

    header_tokens = [t for t in _tokenize_and_filter(header_line, expand_intents=True) if t not in GENERIC_STOPWORDS]
    body_tokens = [t for t in _tokenize_and_filter("\n".join(body_lines) if body_lines else doc, expand_intents=True) if t not in GENERIC_STOPWORDS]

    query_set = set(query_tokens)
    header_set = set(header_tokens)
    body_set = set(body_tokens)

    if not header_set and not body_set:
        return 0.0

    # Header receives double weight to favor intent-bearing headings
    header_overlap = len(query_set & header_set)
    body_overlap = len(query_set & body_set)
    base_score = (2 * header_overlap + body_overlap) / max(1, len(query_set))

    doc_lower = doc.lower()
    q_lower = query.lower()
    header_lower = header_line.lower()

    # --- Phrase-level boosts for arrival/report/check in ---
    arrival_phrases = ["arrive at", "arrive", "arrival", "report", "check in", "check-in"]
    for phrase in arrival_phrases:
        if phrase in q_lower and phrase in doc_lower:
            base_score += 0.4

    # --- Clock-time boost via regex ---
    time_regex = r"\b\d{1,2}(:\d{2})?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])|\b\d{1,2}\s*(a\.?m\.?|p\.?m\.?)(?![a-z])|\b\d{1,2}:\d{2}\b"
    if re.search(time_regex, doc, re.IGNORECASE):
        base_score += 0.5

    # --- Penalty for benefit/admin topics if query is arrival/time intent ---
    penalty_phrases = ["health insurance", "benefits", "pto"]
    if ("arrive" in q_lower or "what time" in q_lower):
        for penalty in penalty_phrases:
            if penalty in header_lower:
                base_score -= 0.5

    # --- Email signature boosts (unchanged) ---
    if 'email' in query_set and 'signature' in query_set:
        has_signature = 'signature' in doc_lower
        has_setup = 'setup' in doc_lower or 'set up' in doc_lower
        has_font = any(kw in doc_lower for kw in ['arial', '10pt', '10 pt', 'font', 'size'])
        
        if has_signature and has_setup:
            base_score += 0.4  # Strong signal
        elif has_signature:
            base_score += 0.25
        
        if has_font:
            base_score += 0.2
        
        field_hits = sum(1 for kw in ['name', 'title', 'phone', 'email', 'website'] if kw in doc_lower)
        if field_hits >= 2:
            base_score += 0.2

    # --- Location richness boosts (unchanged) ---
    if any(kw in doc_lower for kw in ['reception', 'floor', '3rd', 'third', 'front desk', 'frontdesk']):
        location_score = 0
        if "reception" in doc_lower:
            location_score += 0.2
        if "main reception" in doc_lower or "front desk" in doc_lower or "frontdesk" in doc_lower:
            location_score += 0.2
        if "3rd" in doc_lower or "third" in doc_lower:
            location_score += 0.15
        if "floor" in doc_lower:
            location_score += 0.1
        base_score += location_score

    # Bound score deterministically
    return max(0.0, min(2.5, base_score))


def _score_chunk_quality(text: str) -> tuple[float, list[str]]:
    """
    Analyze chunk text for quality signals.
    Returns (penalty_score, reasons).
    Penalty is negative (e.g. -5.0).
    """
    penalty = 0.0
    reasons = []
    
    text_stripped = text.strip()
    if not text_stripped:
        return -10.0, ["quality:empty"]
        
    # 1. Length penalty
    # Very short chunks are often headers or fragments
    if len(text_stripped) < 50:
        penalty -= 2.0
        reasons.append("quality:short_length")
    elif len(text_stripped) < 100:
        penalty -= 0.5
        reasons.append("quality:medium_length")
        
    # 2. Structure penalty (Header-only detection)
    # If it looks like a header (no punctuation at end, short)
    lines = text_stripped.split('\n')
    non_empty_lines = [l for l in lines if l.strip()]
    
    if len(non_empty_lines) == 1:
        line = non_empty_lines[0]
        # If it's short and doesn't end in punctuation, likely a header
        if len(line) < 80 and line[-1] not in ".!?":
            penalty -= 1.5
            reasons.append("quality:header_like")
            
    return penalty, reasons


def _hybrid_rerank_score(query: str, doc: str, vector_distance: float) -> float:
    """
    Combine BM25-style lexical overlap and vector distance for hybrid scoring.
    
    Args:
        query: User's question
        doc: Document chunk text (including headings)
        vector_distance: ChromaDB distance (lower = better)
        
    Returns:
        Combined score (higher = better)
    """

    # --- Query intent detection ---
    q_lower = query.lower()
    is_time_question = (
        ("what time" in q_lower) or
        ("when should i arrive" in q_lower) or
        ("arrive" in q_lower and "time" in q_lower)
    )

    # --- Robust clock time detector ---
    def contains_clock_time(text):
        # Matches times like 8:00 AM, 8:00 A.M., 8 AM, 8AM, 1:00 PM - 2:00 PM
        time_regex = r"\b\d{1,2}(:\d{2})?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])|\b\d{1,2}\s*(a\.?m\.?|p\.?m\.?)(?![a-z])|\b\d{1,2}:\d{2}\b"
        # Also match ranges like "1:00 PM - 2:00 PM"
        range_regex = r"\b\d{1,2}(:\d{2})?\s*(a\.?m\.?|p\.?m\.?)(\s*-\s*\d{1,2}(:\d{2})?\s*(a\.?m\.?|p\.?m\.?))?\b"
        return bool(re.search(time_regex, text, re.IGNORECASE) or re.search(range_regex, text, re.IGNORECASE))

    def contains_duration_only(text):
        # Matches phrases like "within 30 days", "in 2 weeks", "for 1 hour"
        duration_regex = r"\b(within|in|for)\s+\d+\s+(minutes?|hours?|days?|weeks?|months?)\b"
        return bool(re.search(duration_regex, text, re.IGNORECASE))

    # --- Lexical score ---
    lexical_score = _lexical_overlap_score(query, doc)

    # --- Time question special logic ---
    if is_time_question:
        doc_has_clock_time = contains_clock_time(doc)
        doc_has_duration_only = contains_duration_only(doc)
        # Strong bonus for clock time
        if doc_has_clock_time:
            lexical_score = min(1.5, lexical_score + 1.5)
        # Penalty for duration-only (no clock time)
        elif doc_has_duration_only:
            lexical_score = max(0.0, lexical_score - 0.75)

    # --- Penalize chunks with zero lexical overlap (verbs/nouns, not stopwords) ---
    def extract_content_words(text: str) -> set:
        tokens = re.split(r"\s+", re.sub(r"[^A-Za-z0-9]+", " ", text))
        return set(
            t.lower() for t in tokens
            if len(t) > 2 and t.lower() not in STOPWORDS
        )

    query_words = extract_content_words(query)
    doc_words = extract_content_words(doc)
    overlap = query_words & doc_words
    zero_overlap_penalty = 0.15
    has_overlap = len(overlap) > 0

    # --- Vector normalization ---
    # Use 1 / (1 + vector_distance) for (0,1] mapping, higher is better
    vector_score = 1.0 / (1.0 + max(0.0, vector_distance))

    # --- Combine scores deterministically ---
    if is_time_question:
        combined_score = 0.65 * lexical_score + 0.35 * vector_score
    else:
        combined_score = 0.50 * lexical_score + 0.50 * vector_score

    if not has_overlap:
        combined_score = min(combined_score, zero_overlap_penalty)

    # Ensure monotonic and stable in [0,1.5] (since lexical_score can be boosted)
    combined_score = max(0.0, min(1.5, combined_score))
    return combined_score




def _normalize_header(chunk_text: str) -> str:
    """
    Extract and normalize the header (first non-empty line) from a chunk.
    Returns normalized header for deduplication purposes.
    """
    lines = chunk_text.split('\n')
    header = ""
    for line in lines:
        if line.strip():
            header = line.strip()
            break
    
    if not header:
        return ""
    
    # Normalize: lowercase, remove punctuation, collapse whitespace
    normalized = header.lower()
    normalized = re.sub(r'[^a-z0-9\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized




# Query expansion synonym map for common intents
QUERY_EXPANSION_MAP = {
    # Financial/reimbursement related
    'reimburse': ['reimbursement', 'expense report', 'expensify', 'expense'],
    'reimbursed': ['reimbursement', 'expense report', 'expensify', 'expense'],
    'reimbursement': ['expense report', 'expensify', 'expense'],
    
    # Time off / sick leave
    'sick day': ['sick leave', "i'm sick", 'notify manager', 'call in sick', 'illness'],
    'sick leave': ['sick day', "i'm sick", 'notify manager', 'illness'],
    
    # Communication channels
    'email': ['slack', 'communication', 'message', 'contact'],
    'slack': ['email', 'communication', 'message', 'chat'],
    
    # Benefits
    'health insurance': ['medical', 'benefits', 'healthcare', 'coverage'],
    'vacation': ['pto', 'time off', 'paid leave', 'annual leave'],
    'holiday': ['holidays', 'paid holidays', 'company holidays'],
    
    # Onboarding
    'first day': ['onboarding', 'start date', 'orientation', 'new hire'],
    'manager': ['supervisor', 'lead', 'direct report'],
    
    # Remote work
    'remote': ['wfh', 'work from home', 'hybrid'],
    'work from home': ['remote', 'wfh', 'hybrid'],
}


def _expand_query(query: str) -> str:
    """
    Expand query with synonyms and related terms to improve retrieval recall.
    
    Args:
        query: Original user question
        
    Returns:
        Expanded query with added synonym terms
    """
    query_lower = query.lower()
    expansions = []
    
    # Check each expansion trigger
    for trigger, synonyms in QUERY_EXPANSION_MAP.items():
        if trigger in query_lower:
            # Add synonyms that aren't already in the query
            for syn in synonyms:
                if syn.lower() not in query_lower:
                    expansions.append(syn)
    
    # Append expansions to original query
    if expansions:
        expanded = f"{query} {' '.join(expansions)}"
        logger.info("Query expanded: '%s' -> added terms: %s", query[:80], expansions)
        return expanded
    
    return query


# Multi-tenant support: maintain separate collections per tenant
_tenant_collections: Dict[str, Any] = {}

# Embedding cache for frequently queried questions (LRU, max cache)
_embedding_cache: Dict[str, List[float]] = {}


def is_mock_mode() -> bool:
    """
    Check if app is running in mock mode.
    
    Returns True if:
    - RAGIFY_MOCK=1 (explicit mock mode)
    - LLM_PROVIDER=mock (using mock provider)
    - CI=true or APP_MODE=ci (CI environment)
    """
    # Explicit mock mode flag
    if os.getenv("RAGIFY_MOCK", "0") == "1":
        return True
    
    # LLM provider is set to mock
    if os.getenv("LLM_PROVIDER", "").lower() == "mock":
        return True
    
    # CI mode automatically enables mock
    is_ci = (
        os.getenv("CI", "").lower() in ("true", "1", "yes") or
        os.getenv("APP_MODE", "").lower() == "ci"
    )
    return is_ci


def _get_llm_provider() -> LLMProvider:
    global _llm_provider
    if _llm_provider is None:
        provider_name = (os.getenv("LLM_PROVIDER") or "").lower()
        if provider_name == "mock":
            _llm_provider = create_llm_provider(http_client=None)  # mock should not need http
        else:
            http_client = clients.get_http_client()
            _llm_provider = create_llm_provider(http_client=http_client)
    return _llm_provider

def _get_reranker_provider() -> RerankerProvider:
    """Get or create the global reranker provider instance."""
    global _reranker_provider
    if _reranker_provider is None:
        _reranker_provider = create_reranker_provider()
    return _reranker_provider


def _get_embedding_provider():
    """
    Get or create the global embedding provider instance.

    Rules:
    - If EMBEDDING_PROVIDER=mock OR is_mock_mode() => use MockEmbedder and NEVER require HTTP client.
    - If EMBEDDING_PROVIDER=tfidf_test => use TfidfTestEmbedder and NEVER require HTTP client.
    - Otherwise => use RealEmbedder(http_client=clients.get_http_client()).
    """
    global _embedding_provider
    if _embedding_provider is not None:
        return _embedding_provider

    provider = (os.getenv("EMBEDDING_PROVIDER") or "").strip().lower()
    use_mock = (provider == "mock") or (provider not in ("tfidf_test",) and is_mock_mode())
    use_tfidf_test = (provider == "tfidf_test")

    if use_mock:
        # IMPORTANT: mock embedder must not require HTTP client
        try:
            from app.services.embeddings import MockEmbedder
        except ImportError:
            # fallback if class is elsewhere
            from app.services.embeddings import create_embedder
            _embedding_provider = create_embedder()  # must return mock in this mode
            return _embedding_provider

        _embedding_provider = MockEmbedder()
        logger.info("Embedding provider initialized: MockEmbedder (provider=%s, ci/mock=%s)", provider, is_mock_mode())
        return _embedding_provider

    elif use_tfidf_test:
        # TF-IDF test embedder - no HTTP client required
        from app.services.embeddings import TfidfTestEmbedder
        _embedding_provider = TfidfTestEmbedder()
        logger.info("Embedding provider initialized: TfidfTestEmbedder (provider=%s)", provider)
        return _embedding_provider

    # Real embedder path
    from app.services.embeddings import RealEmbedder
    from app.services import clients

    http_client = clients.get_http_client()  # will raise if not initialized (correct in prod)
    _embedding_provider = RealEmbedder(http_client=http_client)
    logger.info("Embedding provider initialized: RealEmbedder (provider=%s)", provider)
    return _embedding_provider

def reset_embedding_provider_for_tests():
    global _embedding_provider
    _embedding_provider = None


def fit_tfidf_test_embedder(texts: List[str]):
    """
    Fit the TF-IDF test embedder on a corpus of texts.

    This should be called once with all documents before any queries when using
    EMBEDDING_PROVIDER=tfidf_test.

    Args:
        texts: List of all text documents/chunks in the corpus
    """
    provider = _get_embedding_provider()
    if not hasattr(provider, 'fit_corpus'):
        logger.warning("Current embedding provider does not support fit_corpus (not TfidfTestEmbedder)")
        return

    logger.info("Fitting TF-IDF test embedder on %d texts", len(texts))
    provider.fit_corpus(texts)
    logger.info("TF-IDF test embedder fitted successfully")
    
def render_prompt_template(prompt_template, *, instruction, history, context, question):
    if callable(prompt_template):
        return prompt_template(
            instruction=instruction,
            history=history,
            context=context,
            question=question,
        )
    elif isinstance(prompt_template, str):
        return prompt_template.format(
            instruction=instruction,
            history=history,
            context=context,
            question=question,
        )
    else:
        raise TypeError(f"prompt_template must be a callable or str, got {type(prompt_template)}")


import asyncio

def get_collection_sync(tenant_id: str = "default"):
    """Synchronous wrapper for get_collection_async. Use only in sync code."""
    return asyncio.get_event_loop().run_until_complete(get_collection_async(tenant_id))

async def get_collection_async(tenant_id: str = "default"):
    # --- Embedding-model/dimension versioned Chroma collection helpers ---
    import chromadb
    from chromadb.config import Settings
    global _tenant_collections
    from app.config import VECTOR_DIR
    embed_provider = _get_embedding_provider()
    provider_name = getattr(embed_provider, "model", None) or getattr(embed_provider, "name", None) or embed_provider.__class__.__name__
    if hasattr(embed_provider, "embedding_dimension"):
        dim = embed_provider.embedding_dimension
    else:
        probe = await embed_texts(["probe"], tenant_id=tenant_id)
        dim = len(probe[0])
    embed_signature = f"{provider_name}__{dim}"
    collection_name = f"documents_{tenant_id}__{embed_signature}"
    cache_key = (tenant_id, embed_signature)
    chroma_client = clients.get_chroma_client()
    if cache_key in _tenant_collections:
        return _tenant_collections[cache_key]
    collection = chroma_client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine", "embed_dim": dim, "embed_provider": provider_name}
    )
    _tenant_collections[cache_key] = collection
    logger.info(f"Initialized collection for tenant: {tenant_id} with embedding {provider_name} dim={dim}")
    return collection


async def embed_texts(texts: List[str], tenant_id: str = "default") -> List[List[float]]:
    """
    Embed a list of texts using the configured embedding provider.
    
    Uses the global embedding provider which can be:
    - MockEmbedder for tests (deterministic SHA-256 vectors)
    - RealEmbedder for production (HTTP calls to Ollama/OpenAI)
    
    Args:
        texts: List of text strings to embed
        tenant_id: Tenant identifier for isolation (used by embedding provider)
        
    Returns:
        List of embedding vectors (one per input text)
    """
    embedder = _get_embedding_provider()
    return await embedder.embed_texts(texts, tenant_id=tenant_id)


async def get_indexed_documents(tenant_id: str) -> List[Dict[str, Any]]:
    """
    Get list of unique documents indexed in ChromaDB for a tenant.
    Returns list of documents with metadata.
    
    Args:
        tenant_id: Tenant identifier
    
    Returns:
        List of document metadata dicts with filename, created_at, status
    """
    if is_mock_mode():
        logger.info("MOCK_MODE: returning empty document list for tenant=%s", tenant_id)
        return []
    
    try:
        # Check if ChromaDB client is available
        try:
            chroma_client = clients.get_chroma_client()
        except RuntimeError:
            logger.debug("ChromaDB client not initialized, skipping document retrieval")
            return []
        
        collection = await get_collection_async(tenant_id)
        
        # Get all documents from the collection with a timeout
        all_items = collection.get()
        
        if not all_items or not all_items.get("metadatas"):
            return []
        
        # Extract unique filenames with their metadata
        seen_files = {}
        for metadata in all_items["metadatas"]:
            # Support both old format (source_file) and new format (filename)
            filename = metadata.get("filename") or metadata.get("source_file", "unknown")
            if filename not in seen_files:
                # Create document entry from metadata
                seen_files[filename] = {
                    "id": metadata.get("doc_id", -1),
                    "filename": filename,
                    "status": "indexed",  # All docs in ChromaDB are indexed
                    "created_at": metadata.get("created_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
                    "updated_at": metadata.get("updated_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
                    "error_message": None
                }
        
        return list(seen_files.values())
        
    except Exception as e:
        logger.warning("Failed to get indexed documents from ChromaDB: %s", e, exc_info=True)
        return []


async def add_documents(tenant_id: str, chunks: List[str], source_filename: str, doc_id: int = -1) -> int:
    """
    Add chunks for a given source file into the tenant-specific Chroma collection.
    Returns number of chunks indexed.
    
    Args:
        tenant_id: Tenant identifier
        chunks: List of text chunks
        source_filename: Original filename
        doc_id: Database document ID for filtering (optional)
    """
    if not chunks:
        return 0

    # If mock mode is enabled, skip actual Chroma operations to avoid external dependencies,
    # unless ALLOW_CHROMA_INDEXING_IN_MOCK is set (for test seeding)
    if is_mock_mode() and os.environ.get("ALLOW_CHROMA_INDEXING_IN_MOCK", "false").lower() != "true":
        logger.info("MOCK_MODE: skipping Chroma indexing for %s (tenant=%s) — returning %d chunks", source_filename, tenant_id, len(chunks))
        return len(chunks)

    logger.info("Indexing %d chunks from %s for tenant %s (doc_id=%s)", len(chunks), source_filename, tenant_id, doc_id)

    collection = await get_collection_async(tenant_id)

    # Remove any existing chunks for this doc so metadata (doc_id) stays in sync
    try:
        if doc_id is not None and doc_id != -1:
            collection.delete(where={"doc_id": {"$eq": doc_id}})
        else:
            collection.delete(where={"filename": {"$eq": source_filename}})
    except Exception as e:
        logger.warning("Failed to delete existing chunks for %s (doc_id=%s): %s", source_filename, doc_id, e)
    
    # Embed all chunks with timing
    t_embed_start = time.time()
    embeddings = await embed_texts(chunks, tenant_id=tenant_id)
    embed_duration = time.time() - t_embed_start
    # Detailed embedding timing already logged in embed_texts
    
    # Include doc_id in the Chroma record IDs to avoid collisions across re-uploads of the same filename
    safe_name = source_filename.replace(" ", "_")
    base_id = f"{doc_id}_{safe_name}" if doc_id is not None and doc_id != -1 else safe_name
    ids = [f"{base_id}_{i}" for i in range(len(chunks))]
    # Store doc_id, filename, and tenant_id metadata for filtering
    metadatas = [
        {
            "tenant_id": tenant_id,
            "source_file": source_filename, 
            "chunk": i,
            "doc_id": doc_id if doc_id is not None else -1,
            "filename": source_filename
        } 
        for i in range(len(chunks))
    ]
    
    # Chroma upsert with timing
    t_upsert = time.time()
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    upsert_duration = time.time() - t_upsert
    log_timing_rag("chroma_upsert", upsert_duration, tenant_id, 
                   num_chunks=len(chunks),
                   upsert_ms=round(upsert_duration * 1000, 2),
                   source_file=source_filename)
    
    # Persist ChromaDB
    chroma_client = clients.get_chroma_client()
    chroma_client.persist()
    
    return len(chunks)




# --- Answer Schema Validators ---


@dataclass
class ValidationResult:
    """Structured result for schema *format* validation.

    ok: overall validity flag
    errors: list of machine-readable error codes / messages
    normalized_text: optional normalized form of the answer
    detected_schema: optional schema inferred from the text when
                     it appears misclassified.
    """

    ok: bool
    errors: List[str]
    normalized_text: Optional[str] = None
    detected_schema: Optional[AnswerSchema] = None


@dataclass
class InvariantResult:
    """Structured result for *content* invariants (schema-agnostic rules)."""

    ok: bool
    errors: List[str]

def _validate_fact_single_response(response: str) -> bool:
    """Validate FACT_SINGLE response: exactly one sentence, no citations.

    Rules:
    - Exactly one non-empty sentence (or single non-empty line)
    - Must NOT contain the canonical refusal phrase
    - Must NOT contain CHUNK_ID-style citation tokens
    """
    import re

    refusal_phrase = "The document does not specify this."

    response = (response or "").strip()
    if not response:
        return False

    # Reject CHUNK_ID-style citations entirely for FACT_SINGLE
    if "(CHUNK_ID=" in response:
        return False

    # Reject embedded refusal phrase for this schema
    if refusal_phrase in response:
        return False

    # Treat as one logical line; split into sentences by punctuation
    sentences = re.split(r"[.!?]+", response)
    sentences = [s.strip() for s in sentences if s.strip()]

    return len(sentences) == 1

def _validate_checklist_procedure_response(response: str) -> bool:
    """Validate CHECKLIST_PROCEDURE response: numbered list, no inline citations.

    Rules:
    - At least one non-empty line.
    - Every line starts with a number and dot (e.g., "1.").
    - Answer text MUST NOT contain CHUNK_ID-style citation tokens; citations
      are carried separately via evidence.
    """
    response = (response or "").strip()
    if not response:
        return False

    if "(CHUNK_ID=" in response:
        return False

    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if not lines:
        return False

    for line in lines:
        # Must start with number followed by dot
        if not re.match(r'^\d+\.', line):
            return False
    return True


def _validate_policy_excerpt_response(response: str) -> bool:
    """Validate POLICY_EXCERPT response: bullet list, no inline citations.

    Rules:
    - At least one non-empty line.
    - Every line starts with "-".
    - Answer text MUST NOT contain CHUNK_ID-style citation tokens.
    """
    response = (response or "").strip()
    if not response:
        return False

    if "(CHUNK_ID=" in response:
        return False

    lines = [line.strip() for line in response.split('\n') if line.strip()]
    if not lines:
        return False

    for line in lines:
        if not line.startswith('-'):
            return False
    return True


def _validate_boolean_specified_response(response: str) -> bool:
    """Validate BOOLEAN_SPECIFIED response: must start with Yes or No, no citations.

    Rules:
    - Response starts with "Yes" or "No" (case-sensitive prefix).
    - MUST NOT contain CHUNK_ID-style citation tokens.
    - "No — the document does not specify this." remains a valid canonical
      refusal *format* for this schema, though invariants may treat the
      refusal phrase specially.
    """
    import re
    response = (response or "").strip()
    if not response:
        return False

    if "(CHUNK_ID=" in response:
        return False

    if response.startswith('Yes'):
        return True

    if response.startswith('No'):
        # Allow canonical refusal wording
        normalized = re.sub(r'[—–-]', ' ', response)
        normalized = re.sub(r'\s+', ' ', normalized)
        if "No the document does not specify this." in normalized:
            return True
        return True

    return False

def _validate_not_found_explicit_response(response: str) -> bool:
    """Validate NOT_FOUND_EXPLICIT response: must equal canonical refusal exactly."""
    return response.strip() == "The document does not specify this."


def _validate_summary_overview_response(response: str) -> bool:
    """Validate SUMMARY_OVERVIEW response.
    
    Rules:
    - Must contain at least 3 bullet points (lines starting with '-').
    - Must NOT contain generic filler phrases.
    - Must NOT contain citations (handled by invariant check, but good to check here too).
    """
    response = (response or "").strip()
    if not response:
        return False
        
    lines = [line.strip() for line in response.split('\n') if line.strip()]
    bullets = [line for line in lines if line.startswith('-')]
    
    if len(bullets) < 3:
        return False
        
    # Check for filler phrases
    filler_phrases = ["typically", "generally", "often", "designed to"]
    response_lower = response.lower()
    for phrase in filler_phrases:
        if phrase in response_lower:
            return False
            
    return True


def _detect_schema_from_text(response: str) -> Optional[AnswerSchema]:
    """Heuristic detection of schema from raw text, for debugging.

    Used only when validation fails, to hint at misclassification.
    """

    text = (response or "").strip()
    if not text:
        return None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # BOOLEAN_SPECIFIED: starts with Yes/No
    lower = text.lower()
    if lower.startswith("yes") or lower.startswith("no"):
        return AnswerSchema.BOOLEAN_SPECIFIED

    # POLICY_EXCERPT: bullet lines starting with '-'
    if lines and all(ln.startswith("-") for ln in lines):
        return AnswerSchema.POLICY_EXCERPT

    # CHECKLIST_PROCEDURE: numbered lines like '1.'
    if lines and all(re.match(r"^\d+\.", ln) for ln in lines):
        return AnswerSchema.CHECKLIST_PROCEDURE

    # FACT_SINGLE: default when we see a CHUNK_ID citation but no
    # clear bullet/numbering structure.
    if "(CHUNK_ID=" in text:
        return AnswerSchema.FACT_SINGLE

    return None


def validate_format_by_schema(response: str, schema: AnswerSchema) -> ValidationResult:
    """Validate *format* of response against the given AnswerSchema."""

    normalized = (response or "").strip()
    errors: List[str] = []

    if schema == AnswerSchema.FACT_SINGLE:
        if not _validate_fact_single_response(normalized):
            errors.append("fact_single_invalid_format")

    elif schema == AnswerSchema.CHECKLIST_PROCEDURE:
        if not _validate_checklist_procedure_response(normalized):
            errors.append("checklist_procedure_invalid_format")

    elif schema == AnswerSchema.POLICY_EXCERPT:
        if not _validate_policy_excerpt_response(normalized):
            errors.append("policy_excerpt_invalid_format")

    elif schema == AnswerSchema.BOOLEAN_SPECIFIED:
        if not _validate_boolean_specified_response(normalized):
            errors.append("boolean_specified_invalid_format")

    elif schema == AnswerSchema.NOT_FOUND_EXPLICIT:
        if not _validate_not_found_explicit_response(normalized):
            errors.append("not_found_explicit_not_canonical")
            
    elif schema == AnswerSchema.SUMMARY_OVERVIEW:
        if not _validate_summary_overview_response(normalized):
            errors.append("summary_overview_invalid")

    # Unknown schema: treat as pass-through but still normalize.

    ok = len(errors) == 0

    detected: Optional[AnswerSchema] = None
    if not ok:
        detected = _detect_schema_from_text(normalized)
        if detected == schema:
            detected = None

    return ValidationResult(
        ok=ok,
        errors=errors,
        normalized_text=normalized,
        detected_schema=detected,
    )


def _validate_response_by_schema(response: str, schema: AnswerSchema) -> ValidationResult:
    """Backwards-compatible alias for format validation.

    Prefer validate_format_by_schema in new code.
    """

    return validate_format_by_schema(response, schema)


def validate_content_invariants(response: str, schema: AnswerSchema) -> InvariantResult:
    """Validate schema-agnostic *content* invariants for a response.

    Examples:
    - Non-NOT_FOUND_EXPLICIT answers must not contain the canonical
      refusal phrase.
    """

    normalized = (response or "").strip()
    errors: List[str] = []

    refusal_phrase = "The document does not specify this."

    if schema != AnswerSchema.NOT_FOUND_EXPLICIT and refusal_phrase in normalized:
        errors.append("refusal_phrase_in_non_not_found")

    ok = len(errors) == 0
    return InvariantResult(ok=ok, errors=errors)


def _is_generic_or_low_overlap(answer: str, evidence_text: str) -> tuple[bool, str]:
    """
    Check if answer is generic or has low grounding overlap.
    Returns (is_generic, reason).
    """
    import re
    answer = answer.lower()
    evidence_text = evidence_text.lower()
    
    # 1. Generic phrases
    generic_phrases = ["typically involves", "generally", "designed to", "often starts with", "usually", "as an ai language model", "i cannot", "i don't have access"]
    for p in generic_phrases:
        if p in answer:
            return True, f"generic_phrase_detected: '{p}'"
        
    # 2. Overlap check
    # Simple tokenization: words > 3 chars
    def tokenize(text):
        return set(w for w in re.findall(r"\w+", text) if len(w) > 3)
        
    ans_tokens = tokenize(answer)
    if not ans_tokens:
        return False, "" # Empty answer, let schema validation handle it
        
    ev_tokens = tokenize(evidence_text)
    
    overlap = len(ans_tokens.intersection(ev_tokens))
    ratio = overlap / len(ans_tokens)
    
    # Threshold: if less than 30% of answer words (len>3) are in evidence, it's suspicious
    if ratio < 0.3:
        return True, f"low_evidence_overlap: {ratio:.2f}"
        
    return False, ""


def _detect_numeric_conflict(question: str, evidence_items: List[Any]) -> Optional[dict]:
    """
    Detects if there are conflicting numeric values for vacation days
    across different source files.
    """
    import re
    
    # 1. Check intent (strict heuristic)
    q_lower = question.lower()
    if "vacation" not in q_lower:
        return None

    years = set(re.findall(r"\b20\d{2}\b", question))
    if years:
        filtered_items = []
        for item in evidence_items:
            snippet = getattr(item, "doc", "") or getattr(item, "snippet", "") or ""
            meta = getattr(item, "meta", {}) or {}
            source_file = meta.get("source_file") or meta.get("filename") or meta.get("file_name") or meta.get("path") or ""
            heading = getattr(item, "heading", "") or ""
            if any(y in snippet or y in source_file or y in heading for y in years):
                filtered_items.append(item)
        if filtered_items:
            evidence_items = filtered_items

    # 2. Extract numbers from evidence
    pattern = re.compile(r"(\d+)\s+vacation\s+days", re.IGNORECASE)
    val_to_sources: dict[int, set[str]] = {}
    
    for item in evidence_items:
        snippet = getattr(item, "doc", "") or getattr(item, "snippet", "")
        if "vacation" not in snippet.lower():
            continue
        chunk_id = getattr(item, "chunk_id", "")
        
        # Extract filename from chunk_id or metadata
        meta = getattr(item, "meta", {}) or {}
        source_file = meta.get("source_file") or meta.get("filename") or meta.get("file_name") or meta.get("path")
        if not source_file and chunk_id:
             parts = chunk_id.rsplit("_", 1)
             if len(parts) >= 2:
                 source_file = parts[0]
        
        if not source_file:
            continue

        matches = pattern.findall(snippet)
        for match in matches:
            try:
                val = int(match)
            except ValueError:
                continue
            if 1 <= val <= 60:
                if val not in val_to_sources:
                    val_to_sources[val] = set()
                val_to_sources[val].add(source_file)
            
    # 3. Analyze conflicts
    if len(val_to_sources) < 2:
        return None

    conflict_sources = sorted({s for sources in val_to_sources.values() for s in sources})
    if len(conflict_sources) < 2:
        return None
        
    # Construct clarification payload
    conflict_values = sorted(val_to_sources.keys())
    year_pattern = re.compile(r"(20\d{2})")
    years = []
    for source in conflict_sources:
        match = year_pattern.search(source)
        if match:
            years.append(match.group(1))
    options = sorted(set(years)) if years else [str(v) for v in conflict_values]
        
    return {
        "pipeline_marker": "CLARIFICATION_REQUIRED",
        "needs_clarification": True,
        "conflict_detected": True,
        "conflict_field": "vacation_days",
        "conflict_values": conflict_values,
        "conflict_sources": conflict_sources,
        "clarification": {
            "type": "TIMEFRAME",
            "question": "Which policy year are you referring to?",
            "options": options
        }
    }


def _construct_schema_correct_answer_from_evidence(
    schema: AnswerSchema,
    evidence_items: List[Any],
) -> Optional[str]:
    """Best-effort schema-correct answer built directly from evidence.

    This is used as a correction path when the LLM incorrectly emits the
    canonical refusal phrase for non-NOT_FOUND_EXPLICIT schemas. It never
    fabricates content: it only rearranges existing evidence snippets into
    the expected schema formats.
    """
    if not evidence_items:
        return None

    refusal_phrase = "The document does not specify this."

    def _first_line(snippet: str) -> str:
        for line in (snippet or "").splitlines():
            s = line.strip()
            if s:
                return s
        return (snippet or "").strip()

    if schema == AnswerSchema.FACT_SINGLE:
        for ev in evidence_items:
            snippet = getattr(ev, "snippet", "") or ""
            for raw_line in snippet.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                # Split by sentence boundaries, but avoid splitting on common abbreviations
                import re
                for candidate in re.split(r"(?<=[.!?])\s+", line):
                    candidate = candidate.strip()
                    if not candidate or refusal_phrase in candidate:
                        continue
                    return candidate

            heading = (getattr(ev, "heading", "") or "").strip()
            if heading and refusal_phrase not in heading:
                return heading

        for ev in evidence_items:
            snippet = (getattr(ev, "snippet", "") or "").strip()
            if snippet:
                return _first_line(snippet) or snippet
            heading = (getattr(ev, "heading", "") or "").strip()
            if heading:
                return heading

        return None


    if schema == AnswerSchema.POLICY_EXCERPT:
        bullets: List[str] = []
        for ev in evidence_items[:3]:
            base = _first_line(getattr(ev, "snippet", ""))
            if not base or refusal_phrase in base:
                continue
            bullets.append(f"- {base}")
        return "\n".join(bullets) if bullets else None

    if schema == AnswerSchema.CHECKLIST_PROCEDURE:
        items: List[str] = []
        for idx, ev in enumerate(evidence_items[:5], start=1):
            base = _first_line(getattr(ev, "snippet", ""))
            if not base or refusal_phrase in base:
                continue
            items.append(f"{idx}. {base}")
        return "\n".join(items) if items else None
        
    if schema == AnswerSchema.SUMMARY_OVERVIEW:
        if not evidence_items:
            return None
            
        # 1. Prefer chunks from the same doc as the top hit
        top_doc_id = getattr(evidence_items[0], "doc_id", None)
        
        # Sort/filter evidence: same doc first, then others
        same_doc_ev = [ev for ev in evidence_items if getattr(ev, "doc_id", None) == top_doc_id]
        other_doc_ev = [ev for ev in evidence_items if getattr(ev, "doc_id", None) != top_doc_id]
        sorted_evidence = same_doc_ev + other_doc_ev
        
        bullets: List[str] = []
        seen_points = set()
        
        for ev in sorted_evidence:
            if len(bullets) >= 8:
                break
                
            snippet = getattr(ev, "snippet", "") or ""
            
            # Extract potential points from snippet
            lines = snippet.split('\n')
            for line in lines:
                line = line.strip()
                if not line or refusal_phrase in line:
                    continue
                    
                candidate = None
                
                # Case 1: Bullet point
                if line.startswith(('-', '*', '•')):
                    candidate = line.lstrip("-*• ").strip()
                    
                # Case 2: Numbered list
                elif re.match(r"^\d+\.\s+", line):
                    candidate = re.sub(r"^\d+\.\s+", "", line).strip()
                    
                # Case 3: ALL CAPS heading or "Heading:" pattern
                elif (line.isupper() and len(line) < 60) or (line.endswith(':') and len(line) < 60):
                    candidate = line.strip(":")
                    
                # Case 4: If we are desperate (few bullets), take the first sentence
                elif len(bullets) < 2 and len(line) > 20 and line[0].isupper():
                     candidate = line
                
                if candidate and candidate not in seen_points:
                    # Strip timestamps if needed (simple heuristic)
                    # e.g. "9:00 AM - Arrival" -> "Arrival"
                    candidate = re.sub(r"^\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\s*[-–]\s*", "", candidate)
                    
                    bullets.append(f"- {candidate}")
                    seen_points.add(candidate)
                    if len(bullets) >= 8: break
            
            # If we didn't get anything from the snippet lines, try the heading/first line
            if len(bullets) < 8:
                heading = getattr(ev, "heading", "")
                first = _first_line(snippet)
                
                # Use heading if available and not seen
                if heading and heading not in seen_points and refusal_phrase not in heading:
                     bullets.append(f"- {heading}")
                     seen_points.add(heading)
                # Else use first line
                elif first and first not in seen_points and refusal_phrase not in first:
                     bullets.append(f"- {first}")
                     seen_points.add(first)

        return "\n".join(bullets) if bullets else None

    # For BOOLEAN_SPECIFIED or unknown schemas, prefer explicit validation failure
    return None


def _select_fact_single_fallback(
    question: str,
    evidence_items: List[Any],
    target_field: Optional[str] = None,
) -> Optional[str]:
    """Pick the best matching sentence/line from evidence for FACT_SINGLE."""
    if not evidence_items:
        return None

    if target_field is None:
        target_field = _detect_benefits_target_field(question)

    if target_field:
        targeted = _extract_targeted_fact_single_from_evidence(
            question,
            evidence_items,
            target_field,
        )
        if targeted:
            return targeted

    refusal_phrase = "The document does not specify this."
    best_line = ""
    best_score = -1.0

    for ev in evidence_items:
        snippet = getattr(ev, "snippet", "") or ""
        for raw_line in snippet.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            import re
            for candidate in re.split(r"(?<=[.!?])\s+", line):
                candidate = candidate.strip()
                if not candidate or refusal_phrase in candidate:
                    continue
                score = _lexical_overlap_score(question, candidate)
                if score > best_score or (score == best_score and len(candidate) > len(best_line)):
                    best_score = score
                    best_line = candidate

    return best_line or None


def repair_answer_by_schema(
    answer_text: str,
    schema: AnswerSchema,
    evidence_items: List[Any],
) -> Optional[str]:
    """Attempt to repair an LLM answer using schema rules and evidence.

    This is invoked before retrying the LLM when the answer fails
    _validate_response_by_schema but does *not* violate the refusal
    invariant. It never fabricates content: any repaired answer is
    constructed from the original answer_text shape and/or evidence.
    """

    answer_text = (answer_text or "").strip()
    if not answer_text:
        return None

    # FACT_SINGLE: if the answer is multi-sentence or contains citations,
    # synthesize a single-sentence answer from top evidence with no
    # inline CHUNK_ID; evidence items themselves serve as citations.
    if schema == AnswerSchema.FACT_SINGLE:
        if _validate_fact_single_response(answer_text):
            return answer_text
        return _construct_schema_correct_answer_from_evidence(
            AnswerSchema.FACT_SINGLE, evidence_items
        )

    # POLICY_EXCERPT: extract 1–3 lines from evidence and format as
    # bullets with CHUNK_ID citations.
    if schema == AnswerSchema.POLICY_EXCERPT:
        if _validate_policy_excerpt_response(answer_text):
            return answer_text
        return _construct_schema_correct_answer_from_evidence(
            AnswerSchema.POLICY_EXCERPT, evidence_items
        )

    # CHECKLIST_PROCEDURE: fall back to the evidence-based constructor
    # to build a numbered checklist if validation failed.
    if schema == AnswerSchema.CHECKLIST_PROCEDURE:
        if _validate_checklist_procedure_response(answer_text):
            return answer_text
        return _construct_schema_correct_answer_from_evidence(
            AnswerSchema.CHECKLIST_PROCEDURE, evidence_items
        )
        
    # SUMMARY_OVERVIEW: fall back to evidence-based constructor
    if schema == AnswerSchema.SUMMARY_OVERVIEW:
        if _validate_summary_overview_response(answer_text):
            return answer_text
        return _construct_schema_correct_answer_from_evidence(
            AnswerSchema.SUMMARY_OVERVIEW, evidence_items
        )

    # BOOLEAN_SPECIFIED: coerce into the canonical
    #   Yes — ... (CHUNK_ID=<id>)
    #   No — ... (CHUNK_ID=<id>)
    # forms when we already have a clear Yes/No leading token. We
    # deliberately avoid emitting the bare refusal phrase; that is
    # reserved for NOT_FOUND_EXPLICIT.
    if schema == AnswerSchema.BOOLEAN_SPECIFIED:
        if _validate_boolean_specified_response(answer_text):
            return answer_text

        if not evidence_items:
            return None
        ev = evidence_items[0]

        lower = answer_text.lower()
        if lower.startswith("yes"):
            # Preserve the original explanation text following "Yes".
            rest = answer_text[3:].lstrip(" —-: ")
            explanation = rest if rest else getattr(ev, "snippet", "").strip()
            repaired = f"Yes — {explanation}".strip()
            return repaired

        if lower.startswith("no"):
            rest = answer_text[2:].lstrip(" —-: ")
            explanation = rest if rest else "the document does not specify this."
            repaired = f"No — {explanation}".strip()
            return repaired

        return None

    # For NOT_FOUND_EXPLICIT and unknown schemas, prefer explicit
    # validation failure rather than silent repair.
    return None

async def query_collection(
    tenant_id: str,
    question: str,
    top_k: int = 4,
    mode: str = "full",
    conversation_history: list[dict] | None = None,
    doc_ids: list[int] | None = None,
    debug: int = 0,
    request_id: str | None = None,
):
    # Determine answer schema from question
    def _determine_answer_schema(q: str) -> AnswerSchema:
        q_lower = q.lower()
        
        # Check for CHECKLIST_PROCEDURE (highest priority)
        if any(phrase in q_lower for phrase in ["what do i do", "steps", "how do i"]):
            return AnswerSchema.CHECKLIST_PROCEDURE
        
        # Check for SUMMARY_OVERVIEW
        if any(word in q_lower for word in ["overview", "process", "summarize", "summary"]) or "what is the onboarding process" in q_lower:
            return AnswerSchema.SUMMARY_OVERVIEW
        
        # Check for POLICY_EXCERPT
        if any(word in q_lower for word in ["policy", "allowed", "required"]):
            return AnswerSchema.POLICY_EXCERPT
        
        # Check for FACT_SINGLE
        if q_lower.startswith(("what time", "where", "who", "what is")):
            return AnswerSchema.FACT_SINGLE
        
        # Check for BOOLEAN_SPECIFIED
        if q_lower.startswith(("is ", "are ")):
            return AnswerSchema.BOOLEAN_SPECIFIED
        
        # Check for NOT_FOUND_EXPLICIT
        if any(word in q_lower for word in ["mentioned", "specified"]):
            return AnswerSchema.NOT_FOUND_EXPLICIT
        
        # Default fallback
        return AnswerSchema.FACT_SINGLE
    
    answer_schema = _determine_answer_schema(question)
    target_field = _detect_benefits_target_field(question)

    # --- Time-arrival and location intent detection ---
    time_arrival_keywords = [
        "when should i arrive",
        "what time do i arrive",
        "arrival time",
        "arrive",
        "check in",
        "check-in",
        "first day",
        "orientation start",
        "report time",
    ]
    location_keywords = [
        "where",
        "location",
        "address",
        "floor",
        "reception",
        "front desk",
        "office",
    ]
    time_regex = r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b"
    q_lower = question.lower()
    is_time_arrival_intent = any(k in q_lower for k in time_arrival_keywords) and (
        "time" in q_lower or "when" in q_lower or "am" in q_lower or "pm" in q_lower
    )
    is_location_intent = any(k in q_lower for k in location_keywords) and "where" in q_lower

    # --- get collection ---
    collection = await get_collection_async(tenant_id)

    # Determine if broad question for higher retrieval
    is_broad = _is_broad_question(question)
    effective_top_k = top_k * 3 if is_broad else top_k  # Retrieve more for broad questions

    # --- embed query and check dimension ---
    question_emb = (await embed_texts([question], tenant_id=tenant_id))[0]
    collection_dim = None
    if hasattr(collection, "metadata") and collection.metadata:
        collection_dim = collection.metadata.get("embed_dim")
    if collection_dim is not None and len(question_emb) != int(collection_dim):
        raise RuntimeError(f"Embedding dimension mismatch: query embedding dim {len(question_emb)} vs collection dim {collection_dim}. Please reindex or purge collections for this embedding model.")


    # --- run chroma query ---
    results = collection.query(
        query_embeddings=[question_emb],
        n_results=max(effective_top_k * 10, effective_top_k),
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    import re
    # --- normalize immediately ---
    hits = _hits_from_chroma(results)
    if doc_ids:
        doc_id_set = set(doc_ids)
        hits = [h for h in hits if h.meta.get("doc_id") in doc_id_set]
    # Compute lexical_score and final_score for every hit
    for h in hits:
        doc_lower = h.doc.lower()
        header_lower = str(h.meta.get("header") or h.meta.get("section") or "").strip().lower()
        contains_clock_time = bool(re.search(time_regex, h.doc, re.IGNORECASE))
        has_arrival_kw = any(kw in doc_lower for kw in time_arrival_keywords) or any(kw in header_lower for kw in time_arrival_keywords)
        h.lexical_score = _lexical_overlap_score(question, h.doc)

        # --- Chunk Quality Penalty ---
        quality_penalty, quality_reasons = _score_chunk_quality(h.doc)

        intent_boost = 0.0
        intent_tags = []

        if quality_penalty != 0:
            intent_boost += quality_penalty
            intent_tags.extend(quality_reasons)
        if is_time_arrival_intent:
            if contains_clock_time and has_arrival_kw:
                intent_boost += 3.0
                intent_tags.append("intent:TIME_ARRIVAL")
                intent_tags.append("clock_time:true")
                intent_tags.append("arrival_terms:true")
            elif contains_clock_time:
                intent_boost += 1.0
                intent_tags.append("intent:TIME_ARRIVAL")
                intent_tags.append("clock_time:true")
            elif has_arrival_kw:
                intent_boost += 0.5
                intent_tags.append("intent:TIME_ARRIVAL")
                intent_tags.append("arrival_terms:true")
            else:
                intent_boost -= 3.0
                intent_tags.append("intent:TIME_ARRIVAL")
                intent_tags.append("penalty:no_clock_time")
        base_score = h.lexical_score if is_time_arrival_intent else _hybrid_rerank_score(question, h.doc, h.dist)
        # Prevent selection of chunks with lexical_score == 0 for this intent if any chunk has lexical_score > 0
        if is_time_arrival_intent and h.lexical_score == 0.0 and any(x.lexical_score > 0 for x in hits):
            intent_boost -= 10.0
            intent_tags.append("penalty:zero_lexical_score")
        h.final_score = base_score + intent_boost
        h.why_selected = intent_tags + (["lexical_overlap"] if h.lexical_score > 0 else [])
    # Sort by final_score DESC, dist ASC, chunk_id ASC
    hits.sort(key=lambda h: (-h.final_score, h.dist, h.chunk_id))

    # --- Conflict Detection (FACT_SINGLE) - Early Check ---
    if answer_schema == AnswerSchema.FACT_SINGLE:
        # Check top 20 hits for conflicts
        conflict_candidates = hits[:20]
        conflict_info = _detect_numeric_conflict(question, conflict_candidates)
        if conflict_info:
            logger.info(f"[RAG] Numeric conflict detected for FACT_SINGLE (early). conflict={conflict_info}")
            
            # Build debug_info
            debug_info = {
                "request_id": request_id,
                "evidence_count": 0,
                "sources_count": 0,
                "refused": False,
                "retrieved_count": len(hits),
                "hits_count": len(hits),
                "answer_schema": answer_schema,
            }
            debug_info.update(conflict_info)
            
            # Return answer_gen that yields ONLY the clarification question once
            async def clarification_gen():
                yield conflict_info["clarification"]["question"]
            
            # Ensure sources/evidence returned are empty lists for clarification
            return clarification_gen(), [], [], "", debug_info

    # --- retrieved_chunks_top20: after normalization, before rerank/dedupe ---
    retrieved_chunks_top20 = None
    if debug >= 1:
        try:
            retrieved_chunks_top20 = []
            for h in hits[:20]:
                try:
                    anchor_type = _get_debug_anchor_type(h.doc)
                    retrieved_chunks_top20.append({
                        "chunk_id": h.chunk_id,
                        "dist": h.dist,
                        "source_file": h.meta.get("source_file") or h.meta.get("filename") or h.meta.get("file_name") or h.meta.get("path"),
                        "header_first_line": _extract_header_first_line(question, h.doc),
                        "contains_clock_time": bool(re.search(time_regex, h.doc, re.IGNORECASE)),
                        "lexical_score": h.lexical_score,
                        "final_score": h.final_score,
                        "intent_signals": h.why_selected,
                        "anchor_type": anchor_type,
                        "anchor_detected": anchor_type is not None,
                    })
                except Exception:
                    retrieved_chunks_top20 = None
                    break
        except Exception:
            retrieved_chunks_top20 = None

    # --- debug_chunks: after rerank/dedupe ---
    debug_chunks = []
    for h in hits:
        debug_chunks.append({
            "chunk_id": h.chunk_id,
            "dist": h.dist,
            "source_file": h.meta.get("source_file") or h.meta.get("filename") or h.meta.get("file_name") or h.meta.get("path"),
            "header_first_line": next((ln.strip() for ln in h.doc.splitlines() if ln.strip()), ""),
            "contains_clock_time": bool(re.search(r"\b\d{1,2}(:\d{2})?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])|\b\d{1,2}\s*(a\.?m\.?|p\.?m\.?)(?![a-z])|\b\d{1,2}:\d{2}\b", h.doc.lower(), re.IGNORECASE)),
            "contains_duration": bool(re.search(r"\b\d+\s*(?:minutes?|hours?|hrs?|hr|mins?)\b", h.doc.lower())),
            "lexical_score": _lexical_overlap_score(question, h.doc),
            "anchor_type": _get_anchor_type(h.doc),
        })

    if not hits:
        async def not_found_gen():
            yield "The document does not specify this."
        debug_info = (
            {"retrieved_count": 0, "selected_count": 0, "chunks": [], "refused": True, "refusal_reason": "NOT_FOUND", "request_id": request_id}
            if debug >= 1 else
            {"refused": True, "refusal_reason": "NOT_FOUND", "request_id": request_id}
        )
        return not_found_gen(), [], [], "", debug_info

    # 1) dedupe — ALL operate on hits (ChunkHit objects), preserve order
    hits_dedup = _dedupe_results(hits)
    hits_dedup = _dedupe_by_header(hits_dedup, max_per_header=1)

    # --- Determine if time-related question ---
    q_lower = question.lower()
    is_time_question = (
        ("what time" in q_lower) or
        ("when should i arrive" in q_lower) or
        ("arrive" in q_lower and "time" in q_lower)
    )
    
    # FIX: Respect top_k from config instead of forcing 5
    internal_k = top_k 

    # FIX: Strictly respect the context budget from config
    min_context_chars = 3000
    max_context_chars = CONTEXT_BUDGET_CHARS if CONTEXT_BUDGET_CHARS else 3000

    def _chunk_context_size(chunk: ChunkHit) -> int:
        return len(chunk.doc) + 80

    def _doc_key(chunk: ChunkHit) -> str:
        return (
            str(chunk.meta.get("doc_id"))
            if chunk.meta.get("doc_id") is not None
            else str(chunk.meta.get("source_file") or chunk.meta.get("filename") or chunk.meta.get("file_name") or chunk.meta.get("path") or chunk.chunk_id)
        )

    def _cap_to_context_budget(chunks: List[ChunkHit]) -> List[ChunkHit]:
        total = 0
        capped: List[ChunkHit] = []
        for h in chunks:
            capped.append(h)
            total += _chunk_context_size(h)
            if total >= max_context_chars:
                break
        return capped

    # 2) final select: include top hit per doc_id/source_file, then fill with MMR until budget
    selected = []
    total_chars = 0
    seen_doc_keys = set()
    seed_hits = []
    for h in hits_dedup:
        key = _doc_key(h)
        if key not in seen_doc_keys:
            seen_doc_keys.add(key)
            seed_hits.append(h)

    for h in seed_hits:
        selected.append(h)
        total_chars += _chunk_context_size(h)

    remaining_candidates = [h for h in hits_dedup if h not in selected]
    if remaining_candidates:
        mmr_ranked = _apply_mmr_selection(remaining_candidates, question_emb, len(remaining_candidates))
        for h in mmr_ranked:
            if total_chars >= max_context_chars:
                break
            selected.append(h)
            total_chars += _chunk_context_size(h)
    
    # ENFORCE intent constraints for time-based questions
    if is_time_arrival_intent:
        def satisfies_intent(chunk):
            doc_lower = chunk.doc.lower()
            header_lower = str(chunk.meta.get("header") or chunk.meta.get("section") or "").strip().lower()
            contains_clock_time = bool(re.search(time_regex, chunk.doc, re.IGNORECASE))
            has_arrival_kw = any(kw in doc_lower for kw in time_arrival_keywords) or any(kw in header_lower for kw in time_arrival_keywords)
            return contains_clock_time and has_arrival_kw
        found = any(satisfies_intent(h) for h in selected)
        if not found:
            async def refuse_gen():
                yield "The document does not specify this."
            debug_info = {
                "retrieved": len(hits),
                "selected": [
                    {"chunk_id": h.chunk_id, "dist": h.dist, "source": h.meta.get("source_file"), "header": h.meta.get("header")}
                    for h in selected
                ],
                "refused": True,
                "refusal_reason": "NO_INTENT_MATCH_FOR_TIME_ARRIVAL",
                "request_id": request_id,
            }
            return refuse_gen(), [], [], "", debug_info
        # Ensure selected_chunks[0] satisfies intent
        if not satisfies_intent(selected[0]):
            for h in selected:
                if satisfies_intent(h):
                    selected = [h] + [x for x in selected if x.chunk_id != h.chunk_id]
                    selected = _cap_to_context_budget(selected)
                    break
            if not satisfies_intent(selected[0]):
                async def refuse_gen():
                    yield "The document does not specify this."
                debug_info = {
                    "retrieved": len(hits),
                    "selected": [
                        {"chunk_id": h.chunk_id, "dist": h.dist, "source": h.meta.get("source_file"), "header": h.meta.get("header")}
                        for h in selected
                    ],
                    "refused": True,
                    "refusal_reason": "NO_INTENT_MATCH_FOR_TIME_ARRIVAL",
                    "request_id": request_id,
                }
                return refuse_gen(), [], [], "", debug_info

    # --- Debug safety: ensure selection is by final_score ---
    if debug >= 1 and selected:
        if selected[0].final_score == 0.0 and any(h.final_score > 0.0 for h in hits_dedup):
            logger.error("[RAG] Selection bug: selected[0] has final_score=0.0 but higher scoring chunks exist. Query: %r", question)
            raise RuntimeError("Selection ordering bug: selected[0] has final_score=0.0 but higher scoring chunks exist.")

    # --- Arrival-time force-inclusion logic ---
    if is_time_arrival_intent:
        # Look at top 20 hits before dedupe
        best_arrival = None
        for h in hits[:20]:
            doc_lower = h.doc.lower()
            has_arrival_kw = any(kw in doc_lower for kw in time_arrival_keywords)
            has_clock_time = bool(re.search(time_regex, h.doc, re.IGNORECASE))
            if has_arrival_kw and has_clock_time:
                if best_arrival is None or h.dist < best_arrival.dist:
                    best_arrival = h
        if best_arrival:
            # Force-include best_arrival in selected, deduping by chunk_id
            selected_ids = {h.chunk_id for h in selected}
            if best_arrival.chunk_id not in selected_ids:
                selected = [best_arrival] + [h for h in selected if h.chunk_id != best_arrival.chunk_id]
                selected = _cap_to_context_budget(selected)
        # Refuse if no selected chunk contains BOTH arrival keyword and clock time
        def has_arrival_and_clock(chunk):
            doc_lower = chunk.doc.lower()
            return any(kw in doc_lower for kw in time_arrival_keywords) and bool(re.search(time_regex, chunk.doc, re.IGNORECASE))
        found = any(has_arrival_and_clock(h) for h in selected)
        if not found:
            async def refuse_gen():
                yield "The document does not specify this."
            debug_info = {
                "retrieved": len(hits),
                "selected": [
                    {"chunk_id": h.chunk_id, "dist": h.dist, "source": h.meta.get("source_file"), "header": h.meta.get("header")}
                    for h in selected
                ],
                "refused": True,
                "refusal_reason": "NO_CLOCK_TIME_FOR_ARRIVAL",
                "request_id": request_id,
            }
            return refuse_gen(), [], [], "", debug_info

    # --- Instrumentation: add final_score and why_selected tags ---
    # Map chunk_id to debug info for after rerank
    chunk_debug_map = {d["chunk_id"]: d for d in debug_chunks}
    for h in hits:
        d = chunk_debug_map.get(h.chunk_id)
        if d is not None:
            d["final_score"] = h.final_score
            d["why_selected"] = h.why_selected if h in selected else []

    # --- PRIMARY EVIDENCE grounding ---
    primary_hit = selected[0] if selected else None
    
    selected_chunk_ids = [h.chunk_id for h in selected] if selected else []
    selected_chunk_headers: dict[str, str] = {}
    context_chunks = []
    if selected:
        # For multi-chunk context, include all selected chunks as evidence
        for i, h in enumerate(selected):
            header = _extract_header_first_line(question, h.doc)
            selected_chunk_headers[h.chunk_id] = header
            source_file = h.meta.get("source_file") or h.meta.get("filename") or h.meta.get("file_name") or h.meta.get("path") or "unknown"
            context_chunks.append(f"[CHUNK_ID={h.chunk_id} SOURCE={source_file} HEADING={header}]\n{h.doc}\n----")
    context_text = "\n\n".join(context_chunks)

    # Rebuild evidence, sources, and context from final selection to keep LLM inputs consistent.
    from app.schemas.query import EvidenceItem
    evidence_items = []
    source_files = []
    context_chunks = []
    selected_chunk_headers = {}
    for h in selected:
        header = _extract_header_first_line(question, h.doc)
        selected_chunk_headers[h.chunk_id] = header

        doc_lower = h.doc.lower()
        anchor_type = None
        if any(kw in doc_lower for kw in ["wifi", "ssid", "password"]):
            anchor_type = "WIFI"
        elif re.search(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", doc_lower, re.IGNORECASE):
            anchor_type = "TIME"

        evidence_items.append(EvidenceItem(
            snippet=h.doc[:400],
            chunk_id=h.chunk_id,
            heading=header,
            doc_id=h.meta.get("doc_id"),
            anchor_type=anchor_type,
            anchor_detected=anchor_type is not None,
        ))

        src = h.meta.get("source_file") or h.meta.get("filename") or h.meta.get("file_name") or h.meta.get("path")
        if src:
            source_files.append(src)

        source_file = src or "unknown"
        context_chunks.append(f"[CHUNK_ID={h.chunk_id} SOURCE={source_file} HEADING={header}]\n{h.doc}\n----")

    seen_src = set()
    source_files = [s for s in source_files if not (s in seen_src or seen_src.add(s))]

    context_text = "\n\n".join(context_chunks)
    selected_chunk_ids = [h.chunk_id for h in selected]

    # --- FIX 1: Define k_final so debug logs don't crash ---
    k_final = len(selected)

    # 5) grounding gate (use selected hits)
    # We compute grounding gate metrics but override refusal for DEV/DEMO modes if needed
    # Prepare the input data in the format grounding.py expects
    grounding_inputs = [(h.doc, h.meta, h.dist) for h in selected]

    # --- FIX 2: Pass selected_chunk_ids as the 3rd argument ---
    should_proceed, refusal_reason, gate_evidence_lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        question,
        grounding_inputs,
        selected_chunk_ids,
        target_field=target_field
    )

    if not should_proceed:
        # OVERRIDE: In DEV or DEMO modes, ignore grounding failures to allow debugging.
        current_mode = str(RAGIFY_MODE).upper()
        if current_mode in ("DEV", "DEMO", "FAST") and not should_proceed:
            logger.warning(f"[RAG] Grounding Gate override active for {current_mode} mode. Proceeding despite low score.")
            should_proceed = True
            refusal_reason = None
        else:
            async def refusal_gen():
                yield "The document does not specify this."
            debug_info = {
                "retrieved": len(hits),
                "selected": [{"chunk_id": h.chunk_id, "dist": h.dist, "source": h.meta.get("source_file"), "header": h.meta.get("header")} for h in selected],
                "refused": True,
                "refusal_reason": refusal_reason,
                "request_id": request_id,
            }
            return refusal_gen(), source_files, evidence_items, context_text, debug_info

    # Guard: If selected chunk does not contain a clock time for time-intent queries, force refusal
    if is_time_arrival_intent and not any(bool(re.search(time_regex, h.doc, re.IGNORECASE)) for h in selected):
        async def refuse_gen():
            yield "The document does not specify this."
        debug_info = {
            "retrieved": len(hits),
            "selected": [
                {"chunk_id": h.chunk_id, "dist": h.dist, "source": h.meta.get("source_file"), "header": h.meta.get("header")}
                for h in selected
            ],
            "refused": True,
            "refusal_reason": "NO_CLOCK_TIME_FOR_ARRIVAL",
            "request_id": request_id,
        }
        return refuse_gen(), [], [], "", debug_info


    # 6) call model strictly with context_text (PRIMARY EVIDENCE only)
    def _llm_prompt_template(*, instruction, history, context, question, answer_schema):
        base_instruction = (
            "You are a helpful assistant. Use the provided EVIDENCE to answer the question. "
            "If the information is not in the evidence, say 'The document does not specify this.' "
            "Provide a direct answer without mentioning the evidence or chunks."
        )

        if answer_schema == AnswerSchema.CHECKLIST_PROCEDURE:
            instruction_str = (
                base_instruction
                + "\n\nOUTPUT FORMAT:\n1. <action>\n2. <action>\n..."
            )
        elif answer_schema == AnswerSchema.POLICY_EXCERPT:
            instruction_str = (
                base_instruction
                + "\n\nOUTPUT FORMAT:\n- <policy statement>\n- <policy statement>\n..."
            )
        elif answer_schema == AnswerSchema.FACT_SINGLE:
            instruction_str = (
                base_instruction
                + "\n\nOUTPUT FORMAT:\n<single sentence answer>"
            )
            if target_field:
                instruction_str += f"\nFocus on {target_field.lower()} benefits."
        elif answer_schema == AnswerSchema.BOOLEAN_SPECIFIED:
            instruction_str = (
                base_instruction
                + "\n\nOUTPUT FORMAT:\nYes <short explanation>.\nOR\nNo. The document does not specify this."
            )
        elif answer_schema == AnswerSchema.SUMMARY_OVERVIEW:
            instruction_str = (
                base_instruction
                + "\n\nOUTPUT FORMAT:\n- <summary point>\n- <summary point>\n..."
            )
        elif answer_schema == AnswerSchema.NOT_FOUND_EXPLICIT:
            instruction_str = (
                "You are a helpful assistant. Answer exactly: 'The document does not specify this.'"
            )
        else:
            instruction_str = base_instruction

        history_str = ""
        if history:
            if isinstance(history, list):
                # Convert conversation history list to a readable string
                history_str = "\n".join([
                    f"{h.get('role', 'user')}: {h.get('content', '')}" if isinstance(h, dict) else str(h)
                    for h in history
                ])
            else:
                history_str = str(history)
        history_section = f"Conversation history:\n{history_str}\n" if history_str else ""
        prompt = (
            f"{instruction_str}\n"
            f"{history_section}"
            f"EVIDENCE:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        return prompt

    # --- WiFi Password Extraction: Deterministic bypass for production ---
    is_wifi_question = any(kw in question.lower() for kw in ["wifi", "wi-fi", "ssid", "network", "wireless", "password"])
    if is_wifi_question and primary_hit:
        import re
        
        # Extract SSID and password from primary_hit.doc using regex
        ssid_match = re.search(r"\bSSID\b\s*[:\-]?\s*([A-Za-z0-9\-_]+)", primary_hit.doc, re.IGNORECASE)
        password_match = re.search(r"\bpassword\b\s*[:\-]?\s*([A-Za-z0-9\-_]+)", primary_hit.doc, re.IGNORECASE)
        
        # Proceed only if we found either SSID or password
        if ssid_match or password_match:
            # Build standardized answer
            answer_parts = []
            if ssid_match:
                answer_parts.append(f"WIFI_SSID: {ssid_match.group(1)}")
            if password_match:
                answer_parts.append(f"WIFI_PASSWORD: {password_match.group(1)}")
            
            standardized_answer = "\n".join(answer_parts)
            
            # Create generator yielding the answer
            async def wifi_extraction_gen():
                yield standardized_answer
            
            # Build debug_info for WiFi extraction
            debug_info = {
                "retrieved": len(hits),
                "selected": [
                    {"chunk_id": h.chunk_id, "dist": h.dist, "source": h.meta.get("source_file"), "header": h.meta.get("header")}
                    for h in selected
                ],
                "refused": False,
                "pipeline_marker": "EXTRACTOR_WIFI",
                "request_id": request_id,
            }
            
            return wifi_extraction_gen(), source_files, evidence_items, context_text, debug_info

    # --- Arrival Time Extraction: Deterministic bypass for production ---
    is_arrival_time_question = any(kw in question.lower() for kw in [
        "what time do i arrive", "what time should i arrive", "arrival time",
        "when do i arrive", "arrive", "check in time", "first day"
    ])

    if is_arrival_time_question and primary_hit:
        import re

        # Prefer times near "arrive" to avoid grabbing lunch/meetings
        arrive_line_match = None
        for line in primary_hit.doc.splitlines():
            if re.search(r"\barrive\b", line, re.IGNORECASE):
                arrive_line_match = line
                break

        time_regex = r"\b\d{1,2}(:\d{2})?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])|\b\d{1,2}\s*(a\.?m\.?|p\.?m\.?)(?![a-z])|\b\d{1,2}:\d{2}\b"
        time_match = None

        if arrive_line_match:
            time_match = re.search(time_regex, arrive_line_match, re.IGNORECASE)

        # Fallback: search full chunk if not found on the arrive line
        if not time_match:
            time_match = re.search(time_regex, primary_hit.doc, re.IGNORECASE)

        if time_match:
            standardized_answer = f"ARRIVAL_TIME: {time_match.group(0).strip()}"

            async def arrival_time_gen():
                yield standardized_answer

            debug_info = {
                "retrieved": len(hits),
                "selected": [
                    {"chunk_id": h.chunk_id, "dist": h.dist, "source": h.meta.get("source_file"), "header": h.meta.get("header")}
                    for h in selected
                ],
                "refused": False,
                "pipeline_marker": "EXTRACTOR_ARRIVAL_TIME",
                "request_id": request_id,
            }

            return arrival_time_gen(), source_files, evidence_items, context_text, debug_info

    # --- Orientation Time Extraction: Deterministic bypass for production ---
    is_orientation_time_question = any(kw in question.lower() for kw in ["orientation time", "when orientation", "orientation start"])
    if is_orientation_time_question and primary_hit:
        import re
        time_regex = r"\b\d{1,2}(:\d{2})?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])|\b\d{1,2}\s*(a\.?m\.?|p\.?m\.?)(?![a-z])|\b\d{1,2}:\d{2}\b"
        time_match = re.search(time_regex, primary_hit.doc, re.IGNORECASE)
        if time_match:
            standardized_answer = f"ORIENTATION_TIME: {time_match.group(0).strip()}"
            
            async def orientation_time_gen():
                yield standardized_answer

            debug_info = {
                "retrieved": len(hits),
                "selected": [
                    {"chunk_id": h.chunk_id, "dist": h.dist, "source": h.meta.get("source_file"), "header": h.meta.get("header")} for h in selected
                ],
                "refused": False,
                "pipeline_marker": "EXTRACTOR_ORIENTATION_TIME",
                "request_id": request_id,
            }

            return orientation_time_gen(), source_files, evidence_items, context_text, debug_info

    # --- Badge Pickup Extraction: Deterministic bypass for production ---
    is_badge_question = any(kw in question.lower() for kw in ["badge", "id card", "security badge", "pickup badge"])
    if is_badge_question and primary_hit and any(kw in primary_hit.doc.lower() for kw in ["badge", "id card", "security"]):
        import re
        # Look for location or process info related to badge pickup
        location_match = re.search(r"(?:at|to|from)\s+([A-Za-z0-9\s\-_]+?)(?:\s+(?:desk|office|reception|security)|$)", primary_hit.doc, re.IGNORECASE)
        if location_match:
            standardized_answer = f"BADGE_PICKUP_LOCATION: {location_match.group(1).strip()}"
        else:
            standardized_answer = "BADGE_PICKUP: Available at reception/security desk"
        
        async def badge_pickup_gen():
            yield standardized_answer

        debug_info = {
            "retrieved": len(hits),
            "selected": [
                {"chunk_id": h.chunk_id, "dist": h.dist, "source": h.meta.get("source_file"), "header": h.meta.get("header")} for h in selected
            ],
            "refused": False,
            "pipeline_marker": "EXTRACTOR_BADGE_PICKUP",
            "request_id": request_id,
            "answer_schema": answer_schema,
        }

        return badge_pickup_gen(), source_files, evidence_items, context_text, debug_info

    # --- Manager Name Extraction: Deterministic bypass for production ---
    is_manager_question = any(kw in question.lower() for kw in ["manager", "supervisor", "boss", "who is my manager"])
    if is_manager_question and primary_hit:
        import re
        # Look for manager/supervisor name patterns
        manager_match = re.search(r"(?:manager|supervisor|boss)\s*(?:is|name)?\s*[:\-]?\s*([A-Za-z\s\-']+)", primary_hit.doc, re.IGNORECASE)
        if manager_match:
            standardized_answer = f"MANAGER_NAME: {manager_match.group(1).strip()}"
            
            async def manager_name_gen():
                yield standardized_answer

            debug_info = {
                "retrieved": len(hits),
                "selected": [
                    {"chunk_id": h.chunk_id, "dist": h.dist, "source": h.meta.get("source_file"), "header": h.meta.get("header")} for h in selected
                ],
                "refused": False,
                "pipeline_marker": "EXTRACTOR_MANAGER_NAME",
                "request_id": request_id,
                "answer_schema": answer_schema,
            }

            return manager_name_gen(), source_files, evidence_items, context_text, debug_info

    # --- Reception Location Extraction: Deterministic bypass for production ---
    is_reception_question = any(kw in question.lower() for kw in ["reception", "front desk", "where reception", "reception location"])
    if is_reception_question and primary_hit and any(kw in primary_hit.doc.lower() for kw in ["reception", "front desk", "lobby"]):
        import re
        # Look for location details
        location_match = re.search(r"(?:reception|front desk)\s+(?:is\s+)?(?:at|on|in)\s+([A-Za-z0-9\s\-_]+?)(?:\s+(?:floor|building|room)|$)", primary_hit.doc, re.IGNORECASE)
        if location_match:
            standardized_answer = f"RECEPTION_LOCATION: {location_match.group(1).strip()}"
        else:
            standardized_answer = "RECEPTION_LOCATION: Main lobby/front desk"
        
        async def reception_location_gen():
            yield standardized_answer

        debug_info = {
            "retrieved": len(hits),
            "selected": [
                {"chunk_id": h.chunk_id, "dist": h.dist, "source": h.meta.get("source_file"), "header": h.meta.get("header")} for h in selected
            ],
            "refused": False,
            "pipeline_marker": "EXTRACTOR_RECEPTION_LOCATION",
            "request_id": request_id,
            "answer_schema": None,  # Will be set by pipeline logic
        }

        return reception_location_gen(), source_files, evidence_items, context_text, debug_info

    # --- Log PRIMARY_HIT (final selection) ---
    primary_hit_final = selected[0] if selected else None
    if primary_hit_final:
        selected_chunk_id = primary_hit_final.chunk_id
        selected_doc_len = len(primary_hit_final.doc)
        contains_password = "ragify-1234" in primary_hit_final.doc.lower()
        contains_guest_ssid = "ragify-guest" in primary_hit_final.doc.lower()
        first_300_chars = primary_hit_final.doc[:300]
        last_300_chars = primary_hit_final.doc[-300:] if len(primary_hit_final.doc) > 300 else primary_hit_final.doc
        logging.info("PRIMARY_HIT request_id=%s selected_chunk_id=%s selected_doc_len=%d contains_password=%s contains_guest_ssid=%s first_300_chars=%r last_300_chars=%r", request_id, selected_chunk_id, selected_doc_len, contains_password, contains_guest_ssid, first_300_chars, last_300_chars)

    numeric_benefits_query = (
        answer_schema == AnswerSchema.FACT_SINGLE
        and target_field
        and _is_numeric_fact_question(question)
    )
    if numeric_benefits_query:
        evidence_texts = [ev.snippet for ev in (evidence_items or []) if getattr(ev, "snippet", None)]
        slot_keywords = SLOT_UNIT_KEYWORDS.get(target_field, [])
        if slot_keywords:
            slot_filtered = [
                text for text in evidence_texts
                if any(kw in text.lower() for kw in slot_keywords)
            ]
            if slot_filtered:
                evidence_texts = slot_filtered
        if evidence_texts:
            sliced_texts = []
            for text in evidence_texts:
                sliced, _ = _slice_text_to_slot(text, target_field)
                if sliced:
                    sliced_texts.append(sliced)
            if sliced_texts:
                evidence_texts = sliced_texts

        consensus_value, has_conflict, _ = extract_numeric_consensus(evidence_texts)
        if consensus_value is not None and not has_conflict:
            formatted = str(int(consensus_value)) if consensus_value.is_integer() else str(consensus_value)
            unit = _format_unit_for_slot(consensus_value, target_field)
            if unit:
                direct_answer = f"According to the documents, the value is {formatted} {unit}."
            else:
                direct_answer = f"According to the documents, the value is {formatted}."

            async def direct_answer_gen():
                yield direct_answer

            debug_info = {
                "retrieved": len(hits),
                "selected": [
                    {"chunk_id": h.chunk_id, "dist": h.dist, "source": h.meta.get("source_file"), "header": h.meta.get("header")}
                    for h in selected
                ],
                "refused": False,
                "pipeline_marker": "EXTRACTOR_DIRECT_HIT",
                "request_id": request_id,
                "answer_schema": answer_schema,
                "topic_slot": target_field,
            }
            return direct_answer_gen(), source_files, evidence_items, context_text, debug_info

    # Call the chat model with the prompt template (instruction/history as strings handled by render_prompt_template)
    logger.info("[%s] Calling LLM with context length: %d chars, first 300: %s, last 300: %s", 
                request_id, len(context_text), context_text[:300], context_text[-300:] if len(context_text) > 300 else context_text)
    answer_gen = _call_chat_model(
        question,
        context_text,
        tenant_id,
        mode=mode,
        conversation_history=conversation_history,
        request_id=request_id,
        prompt_template=lambda **kwargs: _llm_prompt_template(**kwargs, answer_schema=answer_schema),
    )

    # --- FIX: Define missing debug variables ---
    coverage_ok = True
    fallback_used = False
    rewritten_query = None
    # -------------------------------------------

    debug_info = None
    if debug >= 1:
        # Assertion: Check consistency between selected_chunk_ids and context_text
        if selected_chunk_ids and context_text:
            first_sel_id = selected_chunk_ids[0]
            match = re.search(r"\[CHUNK_ID=(.*?)\s+SOURCE=", context_text)
            if match:
                first_ctx_id = match.group(1).strip()
                if first_sel_id != first_ctx_id:
                    logger.error(f"[RAG] Consistency Error: selected[0].chunk_id={first_sel_id} != context_text first chunk={first_ctx_id}")
                    raise AssertionError(f"Consistency Error: selected[0] ({first_sel_id}) != context ({first_ctx_id})")

        selected_chunks_debug = []
        for h in selected:
            chunk_id = h.chunk_id
            dist = h.dist
            source_file = h.meta.get("source_file") or h.meta.get("filename") or h.meta.get("file_name") or h.meta.get("path")
            header_first_line = selected_chunk_headers.get(chunk_id) or _extract_header_first_line(question, h.doc)
            contains_clock_time = bool(re.search(r"\b\d{1,2}(:\d{2})?\s*(a\.?m\.?|p\.?m\.?)(?![a-z])|\b\d{1,2}\s*(a\.?m\.?|p\.?m\.?)(?![a-z])|\b\d{1,2}:\d{2}\b", h.doc.lower(), re.IGNORECASE))
            lexical_score = _lexical_overlap_score(question, h.doc)
            # final_score and why_selected (use debug_chunks if available)
            final_score = None
            why_selected = []
            if debug_chunks:
                for d in debug_chunks:
                    if d.get("chunk_id") == chunk_id:
                        final_score = d.get("final_score") if "final_score" in d else None
                        why_selected = d.get("why_selected") if "why_selected" in d else []
                        break
            selected_chunks_debug.append({
                "chunk_id": chunk_id,
                "dist": dist,
                "source_file": source_file,
                "header_first_line": header_first_line,
                "doc": h.doc,
                "contains_clock_time": contains_clock_time,
                "lexical_score": lexical_score,
                "final_score": final_score,
                "why_selected": why_selected if why_selected is not None else [],
                "anchor_type": _get_debug_anchor_type(h.doc),
                "anchor_detected": _get_debug_anchor_type(h.doc) is not None,
            })
        # Guarantee: selected_count always matches selected_chunks_debug length
        context_length = len(context_text) if context_text else 0
        evidence_count = len(evidence_items) if evidence_items else 0
        debug_info = {
            "hits_count": len(hits),
            "selected_count": len(selected_chunks_debug),
            "total_retrieved": len(hits),
            "k_final": k_final,
            "is_broad": is_broad,
            "selected_chunk_ids": selected_chunk_ids,
            "selected_headings": [selected_chunk_headers.get(h.chunk_id) or _extract_header_first_line(question, h.doc) for h in selected],
            "mmr_lambda": 0.6,
            "selected_by": "mmr",
            "coverage_ok": coverage_ok,
            "fallback_used": fallback_used,
            "rewritten_query": rewritten_query,
            "context_length": context_length,
            "evidence_count": evidence_count,
            "pipeline_marker": "MULTI_CHUNK_PIPELINE",
            "retrieved_chunks_top20": retrieved_chunks_top20 if retrieved_chunks_top20 is not None else [],
            "selected_chunks": selected_chunks_debug,
            "request_id": request_id,
            "answer_schema": answer_schema,
            "target_field": target_field,
            "topic_slot": target_field,
            "top10_scores": [], # Placeholder for now, could be populated
            "grounding_gate": {
                "should_proceed": should_proceed,
                "max_overlap": max_overlap,
                "sum_top3": sum_top3,
                "failed_check": failed_check,
                "evidence_lines_count": len(gate_evidence_lines),
            }
        }
    else:
        context_length = len(context_text) if context_text else 0
        evidence_count = len(evidence_items) if evidence_items else 0
        debug_info = {
            "retrieved": len(hits),
            "selected": [
                {
                    "chunk_id": h.chunk_id,
                    "dist": h.dist,
                    "source": h.meta.get("source_file"),
                    "header": h.meta.get("header"),
                }
                for h in selected
            ],
            "total_retrieved": len(hits),
            "k_final": k_final,
            "is_broad": is_broad,
            "selected_chunk_ids": selected_chunk_ids,
            "selected_headings": [selected_chunk_headers.get(h.chunk_id) or _extract_header_first_line(question, h.doc) for h in selected],
            "mmr_lambda": 0.6,
            "selected_by": "mmr",
            "coverage_ok": coverage_ok,
            "fallback_used": fallback_used,
            "rewritten_query": rewritten_query,
            "context_length": context_length,
            "evidence_count": evidence_count,
            "request_id": request_id,
            "answer_schema": answer_schema,
            "target_field": target_field,
            "topic_slot": target_field,
        }


    # --- Post-check: schema validation with retry + invariant correction path ---
    async def validated_answer_gen():
        """Stream a *validated* answer.

        Flow:
        - Buffer first LLM attempt (no tokens yielded yet).
        - Run format + invariant validation.
        - If invalid, attempt deterministic repair from evidence.
        - If still invalid, run a strict retry; if that also fails,
          return structured failure (no streamed refusal tokens).
        - Once a final answer text is chosen, stream it in fixed-size
          chunks so that the concatenated output equals the final text.
        """

        # 1) Consume first attempt from the model without yielding
        answer_text = ""
        async for chunk in answer_gen:
            answer_text += chunk

        final_answer_text: Optional[str] = None

        # 2) Schema-format validation
        vr = validate_format_by_schema(answer_text, answer_schema)
        is_valid = vr.ok
        # Prefer normalized text for downstream checks/repairs
        answer_text_normalized = vr.normalized_text or answer_text

        # Surface format validation details into debug_info
        if isinstance(debug_info, dict):
            if vr.errors:
                debug_info["schema_validation_errors"] = vr.errors
            if vr.detected_schema is not None:
                debug_info["detected_schema"] = vr.detected_schema

        # 3) Content invariants (schema-agnostic rules)
        inv = validate_content_invariants(answer_text_normalized, answer_schema)
        invariant_violated = not inv.ok
        if isinstance(debug_info, dict) and not inv.ok:
            debug_info["content_invariant_errors"] = inv.errors

        # 4) Generic/Hallucination Check (only if invariants passed)
        is_generic_failure = False
        # Skip generic/overlap check for FACT_SINGLE to prevent false positives on short answers
        if not invariant_violated and answer_schema not in [AnswerSchema.NOT_FOUND_EXPLICIT, AnswerSchema.FACT_SINGLE]:
             combined_evidence_text = "\n".join([item.snippet for item in evidence_items]) if evidence_items else ""
             if _is_generic_or_low_overlap(answer_text_normalized, combined_evidence_text):
                 is_generic_failure = True
                 if isinstance(debug_info, dict):
                     debug_info["generic_answer_detected"] = True
                     debug_info["generic_answer_reason"] = "Low overlap or generic phrases detected"

        misalignment_failure = False
        slot_validation_failed = False

        if answer_schema == AnswerSchema.FACT_SINGLE and target_field:
            misalignment_failure = _fact_single_misaligned_to_target(
                answer_text_normalized,
                target_field,
            )
            if misalignment_failure:
                corrected = _extract_targeted_fact_single_from_evidence(
                    question,
                    evidence_items,
                    target_field,
                    context_text=context_text,
                )
                if corrected:
                    final_answer_text = corrected
                    if isinstance(debug_info, dict):
                        debug_info["target_field_misaligned"] = True
                        debug_info["target_field"] = target_field
                        debug_info["fallback_from_evidence"] = True
                        debug_info["pipeline_marker"] = "EXTRACTOR_FACT_SINGLE"
                        debug_info["final_answer_text_override"] = corrected
                else:
                    if isinstance(debug_info, dict):
                        debug_info["target_field_misaligned"] = True
                        debug_info["target_field"] = target_field
                        debug_info["schema_validation_errors"] = (
                            (debug_info.get("schema_validation_errors") or [])
                            + ["target_field_misaligned"]
                        )
                    is_valid = False

        # FACT_SINGLE numeric answers must include evidence-backed numbers + slot keywords.
        if (
            answer_schema == AnswerSchema.FACT_SINGLE
            and target_field
            and _is_numeric_fact_question(question)
            and not final_answer_text
        ):
            candidate_line, slot_slice_applied = _extract_slot_numeric_line(
                question,
                evidence_items,
                target_field,
                context_text=context_text,
            )
            candidate_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", candidate_line or ""))
            slot_keywords = SLOT_UNIT_KEYWORDS.get(target_field, [])
            answer_lower = answer_text_normalized.lower()
            has_slot_keyword = any(kw in answer_lower for kw in slot_keywords)
            has_candidate_number = any(num in answer_text_normalized for num in candidate_numbers)
            if not (has_slot_keyword and has_candidate_number):
                slot_validation_failed = True
                if isinstance(debug_info, dict):
                    debug_info["topic_slot"] = target_field
                    debug_info["slot_slice_applied"] = slot_slice_applied
                    debug_info["validation_failed_reason"] = "slot_numeric_mismatch"
                if candidate_line:
                    final_answer_text = candidate_line
                    if isinstance(debug_info, dict):
                        debug_info["fallback_from_evidence"] = True
                        debug_info["pipeline_marker"] = "EXTRACTOR_FACT_SINGLE"
                        debug_info["final_answer_text_override"] = candidate_line
                else:
                    is_valid = False

        # 3a) Invariant correction path: do NOT retry automatically when
        # the refusal phrase appears for non-NOT_FOUND_EXPLICIT schemas.
        if invariant_violated:
            if isinstance(debug_info, dict):
                debug_info["invariant_violation_detected"] = True

            corrected = _construct_schema_correct_answer_from_evidence(
                answer_schema, evidence_items
            )
            if corrected:
                # We resolved the invariant locally using evidence only.
                final_answer_text = corrected
                if isinstance(debug_info, dict):
                    debug_info["invariant_violation_resolved"] = True
                    debug_info["invariant_violation_final_action"] = (
                        "CORRECTED_USING_EVIDENCE"
                    )
                    debug_info["corrected_answer_snippet"] = corrected[:200]
                    debug_info["final_answer_text_override"] = corrected
            else:
                fallback_line = None
                if answer_schema == AnswerSchema.FACT_SINGLE and evidence_items:
                    fallback_line = _select_fact_single_fallback(
                        question,
                        evidence_items,
                        target_field=target_field,
                    )
                if fallback_line:
                    final_answer_text = fallback_line
                    if isinstance(debug_info, dict):
                        debug_info["fallback_from_evidence"] = True
                        debug_info["pipeline_marker"] = "EXTRACTOR_FACT_SINGLE"
                        debug_info["refused"] = False
                        debug_info["refusal_reason"] = None
                        debug_info["final_answer_text_override"] = fallback_line
                else:
                    # Could not construct a schema-correct answer from evidence
                    # -> mark validation failure with refusal and reason and
                    # return a structured failure (no streamed refusal tokens).
                    if isinstance(debug_info, dict):
                        debug_info["invariant_violation_resolved"] = False
                        debug_info["invariant_violation_final_action"] = (
                            "VALIDATION_FAILED_FORCED_REFUSAL"
                        )
                        debug_info["validation_failed"] = True
                        debug_info["validation_schema"] = answer_schema.value
                        debug_info["validation_attempts"] = 1
                        debug_info["refused"] = True
                        debug_info["refusal_reason"] = "VALIDATION_FAILED"
                        debug_info["pipeline_marker"] = "FORCED_REFUSAL"
                    # No answer will be streamed; caller inspects debug/decision.
                    return

        # 3b) Pure schema validation failures OR Generic content failures
        if not final_answer_text and not invariant_violated and (not is_valid or is_generic_failure):
            repaired = None
            # Only attempt schema repair if it was a schema failure, not a generic content failure
            if not is_valid and not is_generic_failure:
                repaired = repair_answer_by_schema(
                    answer_text_normalized, answer_schema, evidence_items
                )
            
            if repaired:
                final_answer_text = repaired
                if isinstance(debug_info, dict):
                    debug_info["schema_repair_used"] = True
                    debug_info["schema_repair_source"] = "evidence"
                    debug_info["final_answer_text_override"] = repaired
            else:
                if is_generic_failure:
                    retry_reason = "generic content failure"
                elif misalignment_failure:
                    retry_reason = "target field misalignment"
                elif slot_validation_failed:
                    retry_reason = "slot numeric validation failure"
                else:
                    retry_reason = "schema validation failure"
                logger.warning(
                    f"[RAG] {retry_reason.capitalize()} for {answer_schema.value}, attempting retry. request_id={request_id}"
                )

                retry_instruction = (
                    "I'm sorry, I couldn't find a clear answer in your previous response. "
                    "Please look closely at the EVIDENCE provided below and provide the specific detail requested.\n\n"
                    f"EVIDENCE:\n{context_text}\n\nQuestion: {question}\nAnswer:"
                )

                # Retry with strict prompt (non-streaming)
                retry_gen = _call_chat_model(
                    question,
                    context_text,
                    tenant_id,
                    mode=mode,
                    conversation_history=conversation_history,
                    request_id=f"{request_id}_retry",
                    prompt_template=lambda **kwargs: retry_instruction,
                )

                retry_answer_text = ""
                async for chunk in retry_gen:
                    retry_answer_text += chunk

                # Validate retry response: format + content invariants
                vr_retry = validate_format_by_schema(retry_answer_text, answer_schema)
                retry_is_valid = vr_retry.ok
                inv_retry = validate_content_invariants(retry_answer_text, answer_schema)
                retry_invariant_violated = not inv_retry.ok

                if isinstance(debug_info, dict):
                    if vr_retry.errors:
                        debug_info["schema_validation_errors_retry"] = vr_retry.errors
                    if vr_retry.detected_schema is not None:
                        debug_info["detected_schema_retry"] = vr_retry.detected_schema
                    if inv_retry.errors:
                        debug_info["content_invariant_errors_retry"] = inv_retry.errors

                retry_misaligned = False
                if answer_schema == AnswerSchema.FACT_SINGLE and target_field:
                    retry_misaligned = _fact_single_misaligned_to_target(
                        retry_answer_text,
                        target_field,
                    )
                    if retry_misaligned:
                        corrected_retry = _extract_targeted_fact_single_from_evidence(
                            question,
                            evidence_items,
                            target_field,
                            context_text=context_text,
                        )
                        if corrected_retry:
                            final_answer_text = corrected_retry
                            if isinstance(debug_info, dict):
                                debug_info["target_field_misaligned_retry"] = True
                                debug_info["target_field"] = target_field
                                debug_info["fallback_from_evidence"] = True
                                debug_info["pipeline_marker"] = "EXTRACTOR_FACT_SINGLE"
                                debug_info["final_answer_text_override"] = corrected_retry
                        else:
                            retry_is_valid = False
                            if isinstance(debug_info, dict):
                                debug_info["target_field_misaligned_retry"] = True
                                debug_info["target_field"] = target_field

                if not retry_is_valid or retry_invariant_violated:
                    # Retry also failed: record structured failure and
                    # return without streaming any tokens.
                    fallback_line = None
                    if answer_schema == AnswerSchema.FACT_SINGLE and evidence_items:
                        fallback_line = _select_fact_single_fallback(
                            question,
                            evidence_items,
                            target_field=target_field,
                        )
                    if fallback_line:
                        final_answer_text = fallback_line
                        if isinstance(debug_info, dict):
                            debug_info["fallback_from_evidence"] = True
                            debug_info["pipeline_marker"] = "EXTRACTOR_FACT_SINGLE"
                            debug_info["refused"] = False
                            debug_info["refusal_reason"] = None
                            debug_info["final_answer_text_override"] = fallback_line
                    else:
                        if retry_invariant_violated:
                            logger.error(
                                f"[RAG] Invariant violation persisted on retry for {answer_schema.value}, forcing NOT_FOUND_EXPLICIT. request_id={request_id}"
                            )
                            if isinstance(debug_info, dict):
                                debug_info["invariant_violation_forced_refusal"] = True
                                debug_info["refused"] = True
                                debug_info["original_schema"] = answer_schema.value
                        else:
                            logger.error(
                                f"[RAG] Schema validation failed on retry for {answer_schema.value}, failing cleanly. request_id={request_id}"
                            )
                            if isinstance(debug_info, dict):
                                debug_info["validation_failed"] = True
                                debug_info["validation_schema"] = answer_schema.value
                                debug_info["validation_attempts"] = 2
                        if isinstance(debug_info, dict):
                            debug_info["refused"] = True
                            debug_info["refusal_reason"] = "VALIDATION_FAILED"
                            debug_info["pipeline_marker"] = "FORCED_REFUSAL"
                        return
                else:
                    # Retry succeeded
                    if not final_answer_text:
                        final_answer_text = retry_answer_text
                    if isinstance(debug_info, dict):
                        debug_info["validation_retried"] = True
                        debug_info["validation_schema"] = answer_schema.value

        # 3c) First attempt succeeded with no invariant violation
        if not final_answer_text and not invariant_violated and is_valid:
            final_answer_text = answer_text_normalized
            if isinstance(debug_info, dict):
                debug_info["validation_passed"] = True
                debug_info["validation_schema"] = answer_schema.value

        # Numeric grounding gate after final answer is chosen.
        if final_answer_text and answer_schema == AnswerSchema.FACT_SINGLE and _is_numeric_fact_question(question):
            evidence_texts = [ev.snippet for ev in (evidence_items or []) if getattr(ev, "snippet", None)]
            consensus_value, _, _ = extract_numeric_consensus(evidence_texts)
            if consensus_value is not None:
                lower_answer = final_answer_text.strip().lower()
                refusal_like = (
                    "does not specify" in lower_answer
                    or "does not explicitly contain" in lower_answer
                    or "document does not specify" in lower_answer
                )
                if refusal_like or not validate_numeric_alignment(final_answer_text, consensus_value):
                    formatted = str(int(consensus_value)) if consensus_value.is_integer() else str(consensus_value)
                    final_answer_text = f"Based on the policy documents, the value is {formatted}."
                    logger.warning(
                        "[RAG] Numeric alignment failed; forcing evidence fallback. request_id=%s",
                        request_id,
                    )
                    if isinstance(debug_info, dict):
                        debug_info["validation_failed_reason"] = "numeric_alignment_failed"
                        debug_info["fallback_from_evidence"] = True
                        debug_info["pipeline_marker"] = "EXTRACTOR_EVIDENCE_FALLBACK"
                        debug_info["refused"] = False
                        debug_info["refusal_reason"] = None

        # If we still have no final answer text at this point, nothing to stream.
        if not final_answer_text:
            return

        # 4) Stream the final answer text in fixed-size chunks so that
        # concatenated output equals final_answer_text.
        chunk_size = 80  # between 20 and 100 characters
        text = final_answer_text
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]

    # --- Post-check: if answer includes a clock time but evidence[0].snippet does not, force refusal ---
    async def checked_answer_gen():
        answer_text = ""
        async for chunk in validated_answer_gen():
            answer_text += chunk
            yield chunk
        # Only check if evidence and answer both exist
        if evidence_items and hasattr(evidence_items[0], "snippet"):
            ev_snippet = evidence_items[0].snippet or ""
            # Find all clock times in answer
            answer_times = re.findall(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", answer_text, re.IGNORECASE)
            # Flatten to strings
            answer_times_flat = [" ".join([t[0], t[1]]).strip() if t[1] else t[0] for t in answer_times]
            for t in answer_times_flat:
                if t and t.lower() not in ev_snippet.lower():
                    logging.error(f"[RAG] DEMO SAFETY: Answer contains clock time '{t}' absent from evidence. request_id={request_id}")
                    # Update debug_info to indicate refusal; do not emit forced refusal tokens
                    if isinstance(debug_info, dict):
                        debug_info["refused"] = True
                        debug_info["refusal_reason"] = "time_not_in_evidence"
                    return
    return checked_answer_gen(), source_files, evidence_items, context_text, debug_info


def clear_embedding_cache() -> None:
    """Clear the query embedding cache."""
    global _embedding_cache
    _embedding_cache.clear()
    logger.info("Embedding cache cleared")


async def _call_chat_model(
    question: str,
    context: str,
    tenant_id: str,
    mode: str = "full",
    conversation_history: Optional[List[Dict]] = None,
    validate_before_stream: bool = True,
    request_id: Optional[str] = None,
    prompt_template: Optional[str] = None,
    **kwargs: Any,
) -> AsyncGenerator[str, None]:
    """
    Lightweight replacement of the original _call_chat_model.
    Behaviour:
    - In mock/CI mode, emits deterministic answers extracted from the PRIMARY EVIDENCE section or
      yields the canonical refusal phrase.
    - In normal mode, delegates to generate_answer_stream and enforces simple citation checks.
    This implementation is intentionally conservative and preserves the streamed interface.
    """
    refusal_text = "The document does not specify this."

    # Build simple history string
    history_text = ""
    if conversation_history:
        for msg in conversation_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_text += f"{role}: {msg.get('content','')}\n\n"

    # Simple prompt rendering (allow override)
    if prompt_template:
        prompt = render_prompt_template(prompt_template, instruction="", history=history_text, context=context, question=question)
    else:
        prompt = f"Evidence:\n{context}\n\nQuestion: {question}\nAnswer:"

    # Mock deterministic behaviour
    if is_mock_mode():
        ctx = context or ""
        primary = ctx
        if "PRIMARY EVIDENCE" in ctx:
            primary = ctx.split("PRIMARY EVIDENCE", 1)[1]

        lines = [ln.strip() for ln in primary.splitlines() if ln.strip()]
        # Try time heuristics
        import re
        time_re = re.compile(r"\b(\d{1,2}(:\d{2})?\s*(?:am|pm))\b", re.IGNORECASE)
        for ln in lines:
            if time_re.search(ln):
                # ensure citation present
                if "(chunk_id:" not in ln:
                    # attach first chunk id if available
                    cid_match = re.search(r"chunk_id=([\w\-]+)", context or "")
                    if cid_match:
                        ln = ln + f" (chunk_id:{cid_match.group(1)})"
                yield ln.strip()
                return

        # fallback: keyword overlap
        q = (question or "").lower()
        keywords = [w for w in re.findall(r"[a-z]+", q) if w not in {"what","when","where","is","the","do","i","my","a","an","to","of","and"}]
        best_ln, best_score = None, -1
        for ln in lines:
            score = sum(1 for w in keywords if w in ln.lower())
            if score > best_score:
                best_score = score
                best_ln = ln
        if best_ln and best_score > 0:
            if "(chunk_id:" not in best_ln:
                cid_match = re.search(r"chunk_id=([\w\-]+)", context or "")
                if cid_match:
                    best_ln = best_ln + f" (chunk_id:{cid_match.group(1)})"
            yield best_ln.strip()
            return

        # nothing found -> refusal
        yield refusal_text
        return

    # Non-mock: use provider stream
    provider = _get_llm_provider()
    import re
    allowed_chunk_ids = set(re.findall(r"chunk_id=([\w\-]+)", context or ""))

    async def _citations_valid(text: str) -> bool:
        cited = set(re.findall(r"chunk_id:([\w\-]+)", text or ""))
        return cited.issubset(allowed_chunk_ids)

    if validate_before_stream:
        buf = ""
        async for chunk in generate_answer_stream(prompt=prompt, tenant_id=tenant_id, provider=provider, max_tokens=(MAX_TOKENS_FULL if mode!="fast" else MAX_TOKENS_FAST), timeout=REQUEST_TIMEOUT, validate_fn=None, evidence_text=context, refusal_text=refusal_text, request_id=request_id, chunk_size=75):
            buf += chunk

        # citation enforcement
        if not await _citations_valid(buf):
            yield refusal_text
            return

        final = (buf or "").strip()
        # if refusal phrase present, return canonical refusal only
        if refusal_text in final:
            yield refusal_text
            return

        # remove inline (chunk_id:xxx) occurrences for cleanliness
        cleaned = re.sub(r"\s*\(chunk_id:[\w\-]+\)", "", final)
        yield cleaned
        return

    # streaming path
    streamed = ""
    async for chunk in generate_answer_stream(prompt=prompt, tenant_id=tenant_id, provider=provider, max_tokens=(MAX_TOKENS_FULL if mode!="fast" else MAX_TOKENS_FAST), timeout=REQUEST_TIMEOUT, validate_fn=None, evidence_text=context, refusal_text=refusal_text, request_id=request_id, chunk_size=75):
        streamed += chunk
        # if any invalid citations, stop and refuse
        if not await _citations_valid(streamed):
            yield refusal_text
            return
        yield chunk


def reset_collection(tenant_id: str = "default") -> None:
    """
    Reset the vector store for a specific tenant by clearing all documents and reinitializing.
    Also clears the embedding cache.
    Useful for testing or starting fresh with new documents.
    """
    global _tenant_collections
    
    logger.info("Resetting ChromaDB collection for tenant: %s", tenant_id)
    
    # Clear cache first
    clear_embedding_cache()
    
    # Delete tenant-specific collections (all embed versions) and remove from cache
    chroma_client = clients.get_chroma_client()
    # Remove cached entries for this tenant (keys are tuples: (tenant_id, embed_signature))
    keys_to_remove = [k for k in list(_tenant_collections.keys()) if (isinstance(k, tuple) and k[0] == tenant_id) or (k == tenant_id)]
    for k in keys_to_remove:
        try:
            # Attempt to compute collection name and delete if possible
            if isinstance(k, tuple):
                _, embed_sig = k
                collection_name = f"documents_{tenant_id}__{embed_sig}"
            else:
                collection_name = f"documents_{tenant_id}"
            try:
                chroma_client.delete_collection(collection_name)
                logger.info("Deleted collection: %s", collection_name)
            except Exception:
                logger.debug("Could not delete collection %s (may not exist)", collection_name)
        finally:
            try:
                del _tenant_collections[k]
            except KeyError:
                pass
    
    logger.info("Collection reset complete for tenant: %s", tenant_id)


# Wrapper functions for backward compatibility and convenience
async def index_files(tenant_id: str, chunks: List[str], source_filename: str, doc_id: int = None) -> int:
    """
    Convenience wrapper for add_documents with tenant support.
    """
    return await add_documents(tenant_id, chunks, source_filename, doc_id=doc_id)


async def answer_question(
    tenant_id: str, 
    question: str, 
    top_k: int = 4, 
    mode: str = "full",
    conversation_history: List[Dict] = None,
    doc_ids: List[int] = None,
    debug: int = 0,
    request_id: str = None
) -> Tuple[AsyncGenerator[str, None], List[str], List[str], str, Dict[str, Any], AnswerDecision]:
    """
    Convenience wrapper for query_collection with tenant support.
    
    Args:
        tenant_id: Tenant identifier
        question: User's question
        top_k: Number of chunks to retrieve
        mode: "fast" (concise, max_tokens=50) or "full" (detailed, no token limit)
        conversation_history: Optional list of previous messages for context
        doc_ids: Optional list of document IDs to filter retrieval (document-scoped search)
        debug: Debug level (0=off, 1=detailed diagnostics)
        request_id: Request identifier for tracing (optional)
    """
    # Normalize doc_ids: empty list behaves the same as None
    if doc_ids is not None and len(doc_ids) == 0:
        doc_ids = None

    answer_gen, sources, evidence, context_text, debug_payload = await query_collection(
        tenant_id,
        question,
        top_k,
        mode=mode,
        conversation_history=conversation_history,
        doc_ids=doc_ids,
        debug=debug,
        request_id=request_id,
    )

    # Construct a structured AnswerDecision from debug payload (no substring matching)
    decision_kwargs: Dict[str, Any] = {
        "decision_type": DecisionType.LLM_VALIDATED,
        "refused": False,
        "refusal_reason": None,
        "answer_schema": None,
        "invariant_violation": None,
        "validation_failed": None,
        "final_answer_text": None,
        "validation_meta": None,
    }

    if isinstance(debug_payload, dict):
        refused = bool(debug_payload.get("refused"))
        refusal_reason = debug_payload.get("refusal_reason")
        answer_schema = debug_payload.get("answer_schema")
        pipeline_marker = debug_payload.get("pipeline_marker") or ""

        decision_type = DecisionType.LLM_VALIDATED
        invariant_violation = None
        validation_failed = None

        if pipeline_marker.startswith("EXTRACTOR_"):
            decision_type = DecisionType.EXTRACTED
        elif refused:
            if debug_payload.get("validation_failed"):
                decision_type = DecisionType.VALIDATION_FAILED
                validation_failed = True
            else:
                decision_type = DecisionType.FORCED_REFUSAL
            if debug_payload.get("invariant_violation_forced_refusal"):
                invariant_violation = True

        decision_kwargs.update(
            {
                "decision_type": decision_type,
                "refused": refused,
                "refusal_reason": refusal_reason,
                "answer_schema": answer_schema,
                "invariant_violation": invariant_violation,
                "validation_failed": validation_failed,
            }
        )

    decision = AnswerDecision(**decision_kwargs)

    return answer_gen, sources, evidence, context_text, debug_payload, decision
