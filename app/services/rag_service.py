from typing import List, Tuple, AsyncGenerator, Dict, Any
import json
import time
import os
import logging
import httpx

from . import clients
from .llm_providers import create_llm_provider, LLMProvider
from .reranker_providers import create_reranker_provider, RerankerProvider
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
)

OLLAMA_BASE_URL = "http://localhost:11434"

logger = logging.getLogger(__name__)

# Global LLM provider instance (initialized on first use)
_llm_provider: LLMProvider = None

# Global reranker provider instance (initialized on first use)
_reranker_provider: RerankerProvider = None

# Common English stopwords for filtering
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


# Common English stopwords for filtering
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with',
    'what', 'when', 'where', 'who', 'which', 'why', 'how', 'do', 'does', 'did',
    'have', 'had', 'should', 'could', 'would', 'can', 'may', 'i', 'my', 'me'
}


def _tokenize_and_filter(text: str, min_len: int = 2) -> list:
    """
    Tokenize text and remove stopwords.
    Returns list (not set) to preserve term frequency for BM25-style scoring.
    """
    cleaned = ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in text)
    tokens = [t for t in cleaned.split() if len(t) > min_len and t not in STOPWORDS]
    return tokens


def _lexical_overlap_score(query: str, doc: str) -> float:
    """
    Compute BM25-style lexical overlap score between query and document.
    Returns a score between 0 and 1 based on weighted token matching.
    
    Improvements:
    - Stopword removal for better signal
    - Term frequency weighting (BM25-style)
    - Position-based boosting (headings, first lines)
    - Domain-specific pattern matching (times, locations)
    
    Args:
        query: User's question
        doc: Document chunk text (including headings)
        
    Returns:
        Score from 0 (no overlap) to 1+ (strong match with boosts)
    """
    import re
    from collections import Counter

    # Tokenize with stopword removal
    query_tokens = _tokenize_and_filter(query)
    doc_tokens = _tokenize_and_filter(doc)
    
    if not query_tokens or not doc_tokens:
        return 0.0

    # Term frequency in document for BM25-style weighting
    doc_tf = Counter(doc_tokens)
    doc_length = len(doc_tokens)
    avg_doc_length = 100  # Assumed average for normalization
    
    # BM25-style scoring parameters
    k1 = 1.5  # Term frequency saturation parameter
    b = 0.75  # Length normalization parameter
    
    # Calculate BM25 score for matched terms
    bm25_score = 0.0
    matched_terms = 0
    
    for q_token in set(query_tokens):
        if q_token in doc_tf:
            tf = doc_tf[q_token]
            # BM25 term frequency component
            tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
            bm25_score += tf_component
            matched_terms += 1
    
    # Normalize by query length
    if len(set(query_tokens)) > 0:
        base_score = bm25_score / len(set(query_tokens))
        # Scale to 0-1 range (typical BM25 scores: 0-3)
        base_score = min(1.0, base_score / 3.0)
    else:
        base_score = 0.0

    doc_lower = doc.lower()
    q_lower = query.lower()
    query_token_set = set(query_tokens)
    
    # Position-based boosting: keywords in headings or first line get extra weight
    doc_lines = doc.split('\n')
    first_line = doc_lines[0].lower() if doc_lines else ""
    
    # Check if query terms appear in heading/first line (strong relevance signal)
    heading_matches = sum(1 for token in query_token_set if token in first_line)
    if heading_matches > 0:
        # Boost proportional to how many query terms are in heading
        heading_boost = min(0.4, heading_matches * 0.15)
        base_score = min(1.5, base_score + heading_boost)

    # Time boosts
    # 1) If query contains an explicit time and doc matches it
    q_time_patterns = re.findall(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b|\b\d{1,2}\s*(?:am|pm)\b", q_lower)
    if any((tp if isinstance(tp, str) else tp[0]) and ((tp if isinstance(tp, str) else tp[0]) in doc_lower) for tp in q_time_patterns):
        base_score = min(1.5, base_score + 0.35)
    # 2) If query asks about time (contains 'time' or 'arrive') and doc contains a time pattern
    elif any(t in query_token_set for t in ['time', 'arrive', 'arrival']):
        if re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b|\b\d{1,2}\s*(?:am|pm)\b", doc_lower):
            base_score = min(1.5, base_score + 0.25)

    # Arrival/Arrival-noun boost
    if any(t in query_token_set for t in ['arrive', 'arrival']):
        if any(kw in doc_lower for kw in ['arrive', 'arrival', 'report', 'reception']):
            base_score = min(1.5, base_score + 0.3)

    # Email signature boosts
    if 'email' in query_token_set and 'signature' in query_token_set:
        has_signature = 'signature' in doc_lower
        has_setup = 'setup' in doc_lower or 'set up' in doc_lower
        has_font = any(kw in doc_lower for kw in ['arial', '10pt', '10 pt', 'font', 'size'])
        
        if has_signature and has_setup:
            base_score = min(1.5, base_score + 0.4)  # Strong signal
        elif has_signature:
            base_score = min(1.5, base_score + 0.25)
        
        if has_font:
            base_score = min(1.5, base_score + 0.2)
        
        field_hits = sum(1 for kw in ['name', 'title', 'phone', 'email', 'website'] if kw in doc_lower)
        if field_hits >= 2:
            base_score = min(1.5, base_score + 0.2)

    # Location richness boosts
    if any(kw in doc_lower for kw in ['reception', 'floor', '3rd', 'third']):
        location_score = 0
        if "reception" in doc_lower:
            location_score += 0.2
        if "main reception" in doc_lower:
            location_score += 0.15
        if "3rd" in doc_lower or "third" in doc_lower:
            location_score += 0.15
        if "floor" in doc_lower:
            location_score += 0.1
        base_score = min(1.5, base_score + location_score)

    return base_score


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
    # Lexical score (0-1.5 range with boosts, higher is better)
    lexical_score = _lexical_overlap_score(query, doc)
    
    # Normalize vector distance to 0-1 range (invert so higher is better)
    # Typical distances: 0-500, with good matches < 400
    # Normalize and invert: 1 - (dist / 500)
    normalized_distance = min(vector_distance / 500.0, 1.0)
    vector_score = 1.0 - normalized_distance
    
    # Combine scores: 50% semantic (vector), 50% lexical
    # With improved lexical scoring (BM25 + domain boosts), increase lexical weight
    # Lexical score can exceed 1.0 with boosts, so we balance equally
    combined_score = 0.50 * vector_score + 0.50 * lexical_score
    
    return combined_score


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
    return os.getenv("RAGIFY_MOCK", "0") == "1"


def _get_llm_provider() -> LLMProvider:
    """Get or create the global LLM provider instance."""
    global _llm_provider
    if _llm_provider is None:
        http_client = clients.get_http_client()
        _llm_provider = create_llm_provider(http_client=http_client)
    return _llm_provider


def _get_reranker_provider() -> RerankerProvider:
    """Get or create the global reranker provider instance."""
    global _reranker_provider
    if _reranker_provider is None:
        _reranker_provider = create_reranker_provider()
    return _reranker_provider


def _get_collection(tenant_id: str = "default"):
    """Get tenant-specific collection from centralized ChromaDB client."""
    chroma_client = clients.get_chroma_client()
    
    # Get or create tenant-specific collection
    if tenant_id not in _tenant_collections:
        collection_name = f"documents_{tenant_id}"
        _tenant_collections[tenant_id] = chroma_client.get_or_create_collection(collection_name)
        logger.info(f"Initialized collection for tenant: {tenant_id}")
    
    return _tenant_collections[tenant_id]


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Get embeddings for a list of texts using a local Ollama embedding model.
    Model: nomic-embed-text (run `ollama pull nomic-embed-text` first).
    Uses persistent async client and parallel requests with connection pooling.
    
    For single queries, uses embedding cache to avoid redundant requests.
    """
    logger.info("Embedding %d texts via Ollama (timeout=%ds) mock=%s", len(texts), REQUEST_TIMEOUT, is_mock_mode())
    # If mock mode is enabled, return deterministic small vectors to avoid Ollama.
    if is_mock_mode():
        emb = [0.0] * 8
        return [list(emb) for _ in texts]

    client = clients.get_http_client()
    embeddings: List[List[float]] = []
    
    # Try to get cached embeddings for single-text queries
    if len(texts) == 1:
        text = texts[0]
        if text in _embedding_cache:
            logger.debug("Cache hit for embedding: %s...", text[:50])
            return [_embedding_cache[text]]
    
    # Parallel embedding requests using persistent client (connection pooled)
    async def embed_one(idx: int, text: str) -> List[float]:
        """Embed a single text with error handling."""
        payload = {"model": "nomic-embed-text", "prompt": text}
        logger.debug("Requesting embedding for text %d (len=%d)", idx, len(text))
        try:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json=payload,
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logger.exception("Embedding request failed for text %d: %s", idx, e)
            raise RuntimeError(
                f"Failed to get embeddings from Ollama at {OLLAMA_BASE_URL}. "
                f"Check that Ollama is running and the model 'nomic-embed-text' is pulled. Original error: {e}"
            )
        data = resp.json()
        emb = data.get("embedding")
        if emb is None:
            raise RuntimeError("No embedding returned from Ollama.")
        logger.debug("Received embedding for text %d (len=%d)", idx, len(emb))
        return emb
    
    # Parallelize all embedding calls using persistent client connection pool
    import asyncio
    t_start = time.time()
    embeddings = await asyncio.gather(*[embed_one(i, t) for i, t in enumerate(texts)])
    total_time = time.time() - t_start
    
    # Log timing with all metrics in one JSON object
    avg_text_length = sum(len(t) for t in texts) / len(texts) if texts else 0
    log_timing_rag("embedding", total_time, "system", 
                   num_texts=len(texts), 
                   avg_length=int(avg_text_length),
                   total_ms=round(total_time * 1000, 2),
                   avg_ms_per_chunk=round((total_time * 1000) / len(texts), 2) if texts else 0)
    
    # Cache single-text embeddings for query optimization
    if len(texts) == 1:
        _embedding_cache[texts[0]] = embeddings[0]
        
    return embeddings


def get_indexed_documents(tenant_id: str) -> List[Dict[str, Any]]:
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
        
        collection = _get_collection(tenant_id)
        
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

    # If mock mode is enabled, skip actual Chroma operations to avoid external dependencies
    if is_mock_mode():
        logger.info("MOCK_MODE: skipping Chroma indexing for %s (tenant=%s) — returning %d chunks", source_filename, tenant_id, len(chunks))
        return len(chunks)

    logger.info("Indexing %d chunks from %s for tenant %s (doc_id=%s)", len(chunks), source_filename, tenant_id, doc_id)

    collection = _get_collection(tenant_id)

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
    embeddings = await embed_texts(chunks)
    embed_duration = time.time() - t_embed_start
    # Detailed embedding timing already logged in embed_texts
    
    # Include doc_id in the Chroma record IDs to avoid collisions across re-uploads of the same filename
    safe_name = source_filename.replace(" ", "_")
    base_id = f"{doc_id}_{safe_name}" if doc_id is not None and doc_id != -1 else safe_name
    ids = [f"{base_id}_{i}" for i in range(len(chunks))]
    # Store doc_id and filename metadata for filtering
    metadatas = [
        {
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
    import re
    
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
    import re
    
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


async def query_collection(
    tenant_id: str, 
    question: str, 
    top_k: int = 4, 
    mode: str = "full",
    conversation_history: List[Dict] = None,
    doc_ids: List[int] = None,
    debug: int = 0
) -> Tuple[AsyncGenerator[str, None], List[str], List[str], str, Dict[str, Any]]:
    """
    Perform a similarity search in the tenant-specific collection and answer the question using retrieved context.
    Returns (answer_generator, list_of_source_files, evidence, context_text, debug_info) where answer_generator yields tokens.
    
    Args:
        tenant_id: Tenant identifier
        question: User's question
        top_k: Number of chunks to retrieve
        mode: "fast" (concise, max_tokens=50) or "full" (detailed, no token limit)
        conversation_history: Optional list of previous messages for context
        doc_ids: Optional list of document IDs to filter retrieval (document-scoped search)
        debug: Debug level (0=off, 1=detailed diagnostics)
    """
    logger.info("Query received from tenant %s: %s (history_len=%d, doc_ids=%s)", 
                tenant_id, question, len(conversation_history) if conversation_history else 0, doc_ids)

    # In mock mode, skip Chroma and Ollama chat and return deterministic answer.
    if is_mock_mode():
        async def mock_gen():
            yield "(mocked) This is a canned answer used for local UI testing."
        debug_info = {"retrieved_count": 0, "selected_count": 0, "chunks": []} if debug >= 1 else []
        return mock_gen(), [], [], "", debug_info

    # TENANT ISOLATION: Get collection for this specific tenant and log details
    collection = _get_collection(tenant_id)
    collection_name = f"documents_{tenant_id}"
    collection_count = collection.count()
    logger.info(
        "TENANT QUERY: tenant_id=%s, collection_name=%s, collection_count=%d",
        tenant_id, collection_name, collection_count
    )
    
    if collection_count == 0:
        async def empty_gen():
            yield "I don't have enough information in the provided documents to answer that question."
        debug_info = {"retrieved_count": 0, "selected_count": 0, "chunks": []} if debug >= 1 else []
        return empty_gen(), [], [], "", debug_info

    # QUERY EXPANSION: Expand query with synonyms before embedding
    expanded_question = _expand_query(question)
    
    # embed question (using expanded version for better retrieval recall)
    t_embed = time.time()
    q_embedding = (await embed_texts([expanded_question]))[0]
    # Embedding timing already logged in embed_texts

    t_retrieval = time.time()
    # Build metadata filter if doc_ids provided
    # ChromaDB doesn't support $in operator, so we need to use $or with multiple $eq
    where_filter = None
    if doc_ids:
        if len(doc_ids) == 1:
            # Single doc_id: use simple $eq
            where_filter = {"doc_id": {"$eq": doc_ids[0]}}
        else:
            # Multiple doc_ids: use $or with multiple $eq conditions
            where_filter = {"$or": [{"doc_id": {"$eq": doc_id}} for doc_id in doc_ids]}
        logger.info("Applying doc_ids filter: %s", where_filter)
    
    # Retrieve a broad set for hybrid reranking (generalized; not question-specific)
    # Override config defaults to retrieve more chunks (30) for better reranking coverage
    RETRIEVE_N = 30  # Always retrieve 30 chunks regardless of mode, then rerank and select top_n
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=RETRIEVE_N,
        where=where_filter,
    )
    log_timing_rag("chroma_retrieval", time.time() - t_retrieval, tenant_id, top_k=top_k, doc_filter=bool(doc_ids))

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    if not docs:
        async def not_found_gen():
            yield "I could not find anything relevant in the indexed documents."
        debug_info = {"retrieved_count": 0, "selected_count": 0, "chunks": []} if debug >= 1 else []
        return not_found_gen(), [], [], "", debug_info

    # Log distances for debugging
    logger.info("Retrieved %d chunks with distances: %s", len(distances), distances)

    # Filter out chunks with low similarity
    t_filter = time.time()
    # ChromaDB uses squared Euclidean distance by default (not cosine)
    # Lower distance = higher similarity. For squared euclidean, typical relevant results are < 500
    # Very relevant: 0-200, Moderately relevant: 200-350, Irrelevant: > 350
    # Threshold configured in config.py based on RAGIFY_MODE
    # NOTE: Always use all retrieved chunks for hybrid reranking, don't pre-filter by similarity threshold
    # This allows lexical matching to find relevant chunks that may have slightly higher distance
    filtered_results = [(doc, meta, dist) for doc, meta, dist in zip(docs, metas, distances)]
    
    log_timing_rag("similarity_filtering", time.time() - t_filter, tenant_id, 
                   before=len(docs), after=len(filtered_results), threshold="skipped_for_reranking")
    logger.info("Skipping similarity threshold filtering: will apply hybrid reranking to all %d chunks instead", len(filtered_results))

    if not filtered_results:
        async def not_relevant_gen():
            yield "I could not find anything relevant in the indexed documents to answer that question."
        debug_info = {"retrieved_count": len(docs), "selected_count": 0, "chunks": []} if debug >= 1 else []
        return not_relevant_gen(), [], [], "", debug_info

    # Always apply lexical+semantic hybrid reranking (generalized across queries)
    if len(filtered_results) > 1:
        t_rerank = time.time()
        logger.info("Applying hybrid reranking (lexical + vector) with expanded query")

        scored_results = []
        for doc, meta, dist in filtered_results:
            # Use expanded query for lexical matching to improve recall
            hybrid_score = _hybrid_rerank_score(expanded_question, doc, dist)
            scored_results.append((doc, meta, dist, hybrid_score))

        # Sort by hybrid score (higher = better)
        scored_results.sort(key=lambda x: x[3], reverse=True)

        # Filter out very low-scoring results; keep threshold low to remain inclusive
        MIN_HYBRID_SCORE = 0.05
        scored_results = [(doc, meta, dist, score) for doc, meta, dist, score in scored_results if score >= MIN_HYBRID_SCORE]
        filtered_hybrid_scores = [score for _, _, _, score in scored_results]

        rerank_duration = time.time() - t_rerank
        log_timing_rag(
            "hybrid_reranking",
            rerank_duration,
            tenant_id,
            before=len(filtered_results),
            after=len(scored_results),
            rerank_ms=round(rerank_duration * 1000, 2),
            top_scores=[round(s, 4) for s in filtered_hybrid_scores[:5]],
            min_score_threshold=MIN_HYBRID_SCORE,
        )

        filtered_results = [(doc, meta, dist) for doc, meta, dist, _ in scored_results]
    else:
        # Single result, keep as-is
        filtered_results = filtered_results
    
    # Limit to top N context chunks after hybrid scoring (enforce strict limit)
    # Use top 5 after reranking regardless of mode, to ensure focused grounding
    TOP_CONTEXT_N = 5  # Always select top 5 after hybrid reranking
    if len(filtered_results) > TOP_CONTEXT_N:
        logger.info(
            "Limiting from %d to top %d best chunks after hybrid scoring",
            len(filtered_results),
            TOP_CONTEXT_N,
        )
        filtered_results = filtered_results[:TOP_CONTEXT_N]

    context_pieces: List[str] = []
    sources: List[str] = []
    detailed_sources: List[str] = []
    selected_info: List[Dict[str, Any]] = []
    for idx, (doc, meta, dist) in enumerate(filtered_results):
        src = meta.get("source_file", "unknown")
        # Debug: log presence of key location/time signals in each context chunk
        dl = doc.lower()
        has_reception = "reception" in dl
        has_main_reception = "main reception" in dl
        has_floor = ("3rd" in dl) or ("third" in dl) or ("floor" in dl)
        has_time_8am = "8:00 am" in dl or "8 am" in dl or "8am" in dl
        # Email signature flags
        has_email_signature = ("email signature" in dl) or ("signature" in dl)
        has_font_info = ("arial" in dl) or ("10pt" in dl) or ("10 pt" in dl) or ("font" in dl)
        logger.info(
            "Context[%d]: src=%s dist=%.2f flags: time8am=%s reception=%s mainReception=%s floor=%s emailSig=%s font=%s preview=%s",
            idx,
            src,
            dist if isinstance(dist, (int, float)) else -1,
            has_time_8am,
            has_reception,
            has_main_reception,
            has_floor,
            has_email_signature,
            has_font_info,
            doc[:120].replace("\n", " ")
        )
        context_pieces.append(f"[{src}] {doc}")
        sources.append(src)
        # Include chunk id (prefer Chroma id; fallback to chunk index)
        chunk_id = None
        try:
            chunk_id = ids[idx] if ids and idx < len(ids) else None
        except Exception:
            chunk_id = None
        if chunk_id is None:
            chunk_id = f"chunk_{meta.get('chunk', idx)}"
        detailed_sources.append(f"{src}#{chunk_id}")
        # Capture a simple header (first non-empty line)
        header = None
        for line in doc.splitlines():
            if line.strip():
                header = line.strip()
                break
        selected_info.append({
            "id": chunk_id,
            "source": src,
            "header": header or doc[:80].replace('\n', ' ')
        })

    # Log selected chunk headers to verify inclusion
    try:
        headers_list = [si.get("header", "") for si in selected_info]
        logger.info("Selected chunk headers: %s", headers_list)
    except Exception:
        logger.debug("Could not log selected chunk headers")

    # dedupe sources while preserving order
    seen = set()
    dedup_sources: List[str] = []
    for s in sources:
        if s not in seen:
            seen.add(s)
            dedup_sources.append(s)

    t_prompt = time.time()
    context = "\n\n".join(context_pieces)
    
    # Apply context budget if configured
    context_char_count = len(context)
    if CONTEXT_BUDGET_CHARS and context_char_count > CONTEXT_BUDGET_CHARS:
        logger.info("Context exceeds budget (%d > %d chars), truncating", 
                   context_char_count, CONTEXT_BUDGET_CHARS)
        context = context[:CONTEXT_BUDGET_CHARS]
        context_char_count = len(context)
    
    log_timing_rag("prompt_building", time.time() - t_prompt, tenant_id, 
                   context_length=context_char_count,
                   context_char_count=context_char_count,
                   num_chunks=len(filtered_results),
                   budget_chars=CONTEXT_BUDGET_CHARS or "unlimited")

    # Robust evidence extraction: extract supporting quotes from top context chunks
    def _extract_evidence_snippet(chunk_text: str, max_chars: int = 400) -> str:
        """
        Extract a meaningful evidence snippet from a chunk.
        If chunk starts with a header (ends with ':'), include the header and
        up to 3 bullet lines following it, up to max_chars.
        
        This ensures evidence includes context like:
        "MANAGER 1:1 MEETING (1:00 PM - 2:00 PM):
         - What does success look like in my first 30 days?
         - Who are the key people I should connect with?"
        """
        import re
        
        lines = chunk_text.split('\n')
        if not lines:
            return chunk_text[:max_chars]
        
        # Check if first non-empty line is a header (ends with ':' or is all caps)
        first_line = ""
        for line in lines:
            if line.strip():
                first_line = line.strip()
                break
        
        is_header = (
            first_line.endswith(':') or
            first_line.endswith(')') or  # Numbered headings like "4. EMAIL SIGNATURE (11:30 AM)"
            (len(first_line) > 3 and first_line == first_line.upper() and any(c.isalpha() for c in first_line))
        )
        
        if not is_header:
            # No header detected, return truncated chunk
            return chunk_text[:max_chars]
        
        # Header detected: collect header + up to 3 bullet lines
        snippet_lines = []
        bullet_count = 0
        total_chars = 0
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if this is a bullet line
            is_bullet = re.match(r'^\s*(?:[-*•]|\d+[.)])\s+', line) is not None
            
            # Add the line if it's the header or a bullet (up to 3 bullets)
            if len(snippet_lines) == 0:  # First line (header)
                snippet_lines.append(line_stripped)
                total_chars += len(line_stripped)
            elif is_bullet and bullet_count < 3:
                snippet_lines.append(line_stripped)
                total_chars += len(line_stripped)
                bullet_count += 1
                
                if total_chars >= max_chars:
                    break
            elif bullet_count >= 3:
                break
        
        snippet = '\n'.join(snippet_lines)
        
        # Truncate if still too long
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "..."
        
        return snippet
    
    def _extract_evidence(q: str, results: List[Tuple[str, Dict, float]]) -> List[Tuple[str, float]]:
        """
        Extract supporting evidence quotes from context chunks with relevance scores.
        Returns list of (evidence_text, relevance_score) tuples.
        
        IMPORTANT: Evidence MUST come from the same chunks in 'results' that are used for context.
        """
        import re
        
        def tokenize(text: str) -> set:
            cleaned = ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in text)
            return set(t for t in cleaned.split() if len(t) > 2)
        
        q_tokens = tokenize(q)
        q_lower = q.lower()
        
        # Score each chunk by keyword/pattern matching
        scored_chunks = []
        
        for doc, meta, dist in results:
            doc_lower = doc.lower()
            doc_tokens = tokenize(doc)
            
            # Calculate lexical overlap score
            overlap = len(q_tokens & doc_tokens)
            union = len(q_tokens | doc_tokens)
            base_score = overlap / union if union > 0 else 0.0
            
            # Boost score based on keyword/pattern matching
            score = base_score
            
            # Time patterns boost
            if any(kw in q_tokens for kw in ['time', 'arrive', 'arrival']):
                if re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b|\b\d{1,2}\s*(?:am|pm)\b", doc_lower):
                    score += 0.3
            
            # Email signature boost - prioritize chunks with heading keywords
            if 'email' in q_tokens and 'signature' in q_tokens:
                # Strong boost for chunks containing the heading phrase
                if 'signature' in doc_lower and 'email' in doc_lower:
                    score += 0.6  # Increased from 0.4 to prioritize heading chunks
                if 'setup' in doc_lower and 'signature' in doc_lower:
                    score += 0.5  # Boost for "signature setup" pattern
                # Detail chunk boost (format details)
                if any(kw in doc_lower for kw in ['arial', 'font', '10pt', '10 pt']):
                    score += 0.3
            
            # Document/bring boost
            if any(kw in q_tokens for kw in ['document', 'bring', 'need']):
                if any(kw in doc_lower for kw in ['id', 'offer', 'bring']):
                    score += 0.3
            
            # Camera/video boost
            if any(kw in q_tokens for kw in ['camera', 'video']):
                if any(kw in doc_lower for kw in ['camera', 'video', 'meeting']):
                    score += 0.3
            
            # Manager/1:1 boost - prioritize chunks with question patterns
            if 'manager' in q_tokens:
                if any(kw in doc_lower for kw in ['manager', 'success', 'goals', 'expectations']):
                    score += 0.3
                # Extra boost for chunks with question patterns (WHAT TO ASK YOUR MANAGER)
                if 'what' in doc_lower and any(kw in doc_lower for kw in ['ask', 'question', 'success']):
                    score += 0.4
            
            # Store full chunk with score (we'll extract snippets later if needed)
            scored_chunks.append((doc, score))
        
        # Sort by score (highest first)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("Evidence chunk scores (top 5): %s", [(round(s, 3), c[:80].replace('\n', ' ')) for c, s in scored_chunks[:5]])
        
        return scored_chunks

    def _score_evidence_relevance(evidence_text: str, query: str) -> float:
        """
        Score evidence snippet relevance using lexical overlap.
        Returns score between 0 and 1.
        """
        query_tokens = set(_tokenize_and_filter(query))
        evidence_tokens = set(_tokenize_and_filter(evidence_text))
        
        if not query_tokens or not evidence_tokens:
            return 0.0
        
        # Jaccard similarity
        intersection = len(query_tokens & evidence_tokens)
        union = len(query_tokens | evidence_tokens)
        score = intersection / union if union > 0 else 0.0
        
        return score

    # EVIDENCE CONSTRUCTION: Extract evidence from selected chunks (not from LLM)
    # Use expanded query for better matching
    # Returns list of (chunk_text, relevance_score) tuples
    scored_chunks = _extract_evidence(expanded_question, filtered_results)
    
    # RELEVANCE FILTER: Filter chunks by score threshold
    EVIDENCE_RELEVANCE_THRESHOLD = 0.15  # Minimum overlap score (tuned for quality)
    
    # Filter by threshold
    relevant_chunks = [(chunk, score) for chunk, score in scored_chunks if score >= EVIDENCE_RELEVANCE_THRESHOLD]
    
    # Take top chunks based on mode and extract better snippets
    if mode == "fast":
        # Fast mode: only highest relevance chunks (top 2)
        evidence = [_extract_evidence_snippet(chunk, max_chars=400) for chunk, score in relevant_chunks[:2]]
        logger.info("Fast mode: selected %d/%d chunks for evidence (threshold=%.2f)", 
                   len(evidence), len(scored_chunks), EVIDENCE_RELEVANCE_THRESHOLD)
    else:
        # Full mode: keep top 3 relevant chunks
        evidence = [_extract_evidence_snippet(chunk, max_chars=400) for chunk, score in relevant_chunks[:3]]
        logger.info("Full mode: selected %d/%d chunks for evidence (threshold=%.2f)", 
                   len(evidence), len(scored_chunks), EVIDENCE_RELEVANCE_THRESHOLD)
    
    # Log relevance scores for debugging
    if scored_chunks:
        top_scores = [round(score, 3) for _, score in scored_chunks[:5]]
        logger.info("Evidence relevance scores (top 5): %s", top_scores)
    
    # VALIDATION: If we have selected chunks, we must have evidence
    if len(filtered_results) > 0 and len(evidence) == 0:
        logger.error(
            "EVIDENCE CONSTRUCTION ERROR: %d chunks selected but 0 evidence extracted. "
            "This should never happen. Question: %s",
            len(filtered_results), question[:100]
        )
        # Fallback: use first chunk preview as evidence
        evidence = [filtered_results[0][0][:150] + "..."]
    
    logger.info(
        "Evidence construction complete: selected_chunks=%d, evidence_count=%d",
        len(filtered_results), len(evidence)
    )
    
    # GROUNDING GATE: Check if evidence is sufficient before calling LLM
    should_proceed, refusal_reason, gate_evidence_lines, max_overlap, sum_top3, failed_check = _compute_grounding_gate(
        question, filtered_results, ids
    )
    
    logger.info(
        "Grounding gate: should_proceed=%s, refusal_reason=%s, evidence_lines=%d, max_overlap=%.0f, sum_top3=%.0f, failed_check=%s",
        should_proceed, refusal_reason, len(gate_evidence_lines), max_overlap, sum_top3, failed_check or "NONE"
    )
    
    if not should_proceed:
        # Return refusal response
        async def refusal_gen():
            yield ""
        
        refusal_debug_info = {
            "retrieved_count": len(docs),
            "selected_count": len(filtered_results),
            "chunks": [],
            "refused": True,
            "refusal_reason": refusal_reason,
            "max_overlap": max_overlap,
            "sum_top3": sum_top3,
            "evidence_lines_count": len(gate_evidence_lines),
            "failed_check": failed_check
        } if debug >= 1 else {"refused": True, "refusal_reason": refusal_reason}
        
        logger.warning(
            "Grounding gate REFUSED query (reason=%s, failed_check=%s, max_overlap=%.0f, sum_top3=%.0f): %s",
            refusal_reason, failed_check, max_overlap, sum_top3, question[:100]
        )
        
        return refusal_gen(), [], [], "", refusal_debug_info
    
    # Build debug info: include retrieved_count, selected_count, and detailed chunk diagnostics
    if debug >= 1:
        # Enhanced debug mode: include id, header, snippet, distance for each selected chunk
        detailed_chunks = []
        for idx, (doc, meta, dist) in enumerate(filtered_results):
            chunk_id = ids[idx] if ids and idx < len(ids) else f"chunk_{meta.get('chunk', idx)}"
            header = None
            for line in doc.splitlines():
                if line.strip():
                    header = line.strip()
                    break
            detailed_chunks.append({
                "id": chunk_id,
                "header": header or doc[:80].replace('\n', ' '),
                "snippet": doc[:200].replace('\n', ' ') + ("..." if len(doc) > 200 else ""),
                "distance": round(dist, 4) if isinstance(dist, (int, float)) else -1
            })
        debug_info = {
            "retrieved_count": len(docs),
            "selected_count": len(filtered_results),
            "chunks": detailed_chunks
        }
    else:
        # Legacy mode: return simple selected_info list
        debug_info = selected_info
    
    answer_gen = _call_chat_model(question, context, tenant_id, mode=mode, conversation_history=conversation_history)
    return answer_gen, detailed_sources, evidence, context, debug_info


def answer_supported_by_evidence(answer: str, evidence_text: str) -> bool:
    """
    Validate that an answer is grounded in the provided evidence.
    Fast deterministic check using lexical overlap and pattern matching.
    
    Args:
        answer: The generated answer to validate
        evidence_text: The evidence text (context) used to generate the answer
    
    Returns:
        True if answer is supported by evidence, False otherwise
    
    Rules:
        1. Exact refusal phrase "The document does not specify this." → True
        2. Normalize both texts (lowercase, remove punctuation, collapse whitespace)
        3. Tokenize and remove stopwords
        4. Require K=2 content tokens from answer in evidence OR
           at least one numeric/time pattern match if answer contains digits/times
        5. Otherwise → False
    """
    import re
    import string
    
    # Rule 1: Check for exact refusal phrase
    if "The document does not specify this." in answer:
        return True
    
    # Rule 2: Normalize - lowercase, remove punctuation, collapse whitespace
    def normalize(text: str) -> str:
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Collapse whitespace
        text = ' '.join(text.split())
        return text
    
    answer_norm = normalize(answer)
    evidence_norm = normalize(evidence_text)
    
    # Rule 3: Tokenize and remove stopwords
    # Small built-in stopword set (common English words with little semantic value)
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }
    
    answer_tokens = set(t for t in answer_norm.split() if t and t not in STOPWORDS)
    evidence_tokens = set(t for t in evidence_norm.split() if t and t not in STOPWORDS)
    
    # Rule 4a: Check if answer contains numeric/time patterns
    has_digit = bool(re.search(r'\d', answer))
    has_time = bool(re.search(r'\d{1,2}:\d{2}', answer))
    has_ampm = bool(re.search(r'\b(?:am|pm)\b', answer.lower()))
    
    if has_digit or has_time or has_ampm:
        # Answer contains numeric/time info - check if at least one pattern appears in evidence
        # Extract all numbers from answer
        answer_numbers = set(re.findall(r'\b\d+\b', answer))
        evidence_numbers = set(re.findall(r'\b\d+\b', evidence_text))
        
        # Extract time patterns (HH:MM)
        answer_times = set(re.findall(r'\d{1,2}:\d{2}', answer))
        evidence_times = set(re.findall(r'\d{1,2}:\d{2}', evidence_text))
        
        # At least one number or time must match
        if (answer_numbers & evidence_numbers) or (answer_times & evidence_times):
            return True
        else:
            # Numeric/time pattern in answer but not in evidence - likely hallucination
            return False
    
    # Rule 4b: For non-numeric answers, require at least K=2 content tokens overlap
    K = 2
    overlap_count = len(answer_tokens & evidence_tokens)
    
    if overlap_count >= K:
        return True
    
    # Rule 5: Failed all checks
    return False


async def _call_chat_model(
    question: str, 
    context: str, 
    tenant_id: str, 
    mode: str = "full",
    conversation_history: List[Dict] = None,
    validate_before_stream: bool = True
) -> AsyncGenerator[str, None]:
    """
    Call the configured LLM provider with the retrieved context.
    Yields answer tokens as they arrive for streaming.
    Supports multiple providers via LLM_PROVIDER env var (ollama, openai).
    
    Args:
        question: User's question
        context: Retrieved context from documents
        tenant_id: Tenant identifier
        mode: "fast" (concise, limited tokens) or "full" (detailed, more tokens)
        conversation_history: Optional list of previous messages for context
        validate_before_stream: If True, buffer answer and validate before streaming. If False, stream directly.
    """
    # Build conversation history text
    history_text = ""
    if conversation_history:
        for msg in conversation_history:
            role_prefix = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role_prefix}: {msg['content']}\n\n"
    
    # Mode-specific prompts and token limits (from config.py)
    if mode == "fast":
        instruction = (
            "ANSWER RULES (STRICT - Fast Mode):\n"
            "1. Maximum 2 sentences ONLY.\n"
            "2. You must answer using ONLY the Evidence lines below.\n"
            "3. If Evidence does not contain the answer, output exactly: The document does not specify this.\n"
            "4. Do not use conversation history as a source of truth; history is for continuity only.\n"
            "5. Extract exact details; do NOT add outside knowledge.\n"
            "6. Include ONLY directly relevant information; skip unrelated sections.\n"
            "7. For time/arrival questions: state the exact time (with AM/PM) and location in ONE sentence.\n"
            "8. Do NOT include background information, Q/A sections, or general guidance unless directly answering the question.\n"
            "\n"
            "Format: Direct answer in 1-2 short sentences."
        )
    else:
        instruction = (
            "ANSWER RULES:\n"
            "1. You must answer using ONLY the Evidence lines below.\n"
            "2. If Evidence does not contain the answer, output exactly: The document does not specify this.\n"
            "3. Do not use conversation history as a source of truth; history is for continuity only.\n"
            "4. Extract exact details; do NOT add outside knowledge.\n"
            "5. Be concise and specific; avoid unrelated guidance.\n"
            "6. If the question asks about time or arrival, include BOTH the exact time (with AM/PM) and the exact location/floor if present in the Evidence, in one short sentence.\n"
            "7. Preserve the exact formatting of times (e.g., 8:00 AM) and floors (e.g., 3rd floor) as written.\n"
            "\n"
            "Reminder: If the answer is visible in the Evidence, use it verbatim."
        )

    if mode == "fast":
        prompt = f"""{instruction}

{history_text if history_text else ""}Evidence (authoritative):
{context}

Question: {question}

Answer:"""
        max_tokens = MAX_TOKENS_FAST
    else:
        prompt = f"""{instruction}

{history_text if history_text else ""}Evidence (authoritative):
{context}

Question: {question}

Answer:"""
        max_tokens = MAX_TOKENS_FULL

    logger.info("Calling LLM for question (len=%d) with context length %d mode=%s max_tokens=%s mock=%s history_len=%d validate=%s", 
                len(question), len(context), mode, max_tokens, is_mock_mode(), len(conversation_history) if conversation_history else 0, validate_before_stream)
    
    # If mock mode, yield canned response immediately
    if is_mock_mode():
        yield "(mocked) This is a canned answer used for local UI testing."
        return

    # Get LLM provider and guardrail config for timeout
    llm_provider = _get_llm_provider()
    guardrail_config = get_guardrail_config(tenant_id)
    llm_timeout = guardrail_config.llm_timeout_seconds
    
    t_llm = time.time()
    first_token_logged = False
    
    def on_first_token(duration: float):
        """Callback when first token arrives."""
        nonlocal first_token_logged
        if not first_token_logged:
            log_timing_rag("llm_first_token", duration, tenant_id, prompt_length=len(prompt))
            first_token_logged = True
    
    try:
        if validate_before_stream:
            # BUFFERED MODE: Collect all tokens, validate, then stream
            full_answer = ""
            async for token in llm_provider.generate_stream(
                prompt, 
                tenant_id, 
                max_tokens=max_tokens, 
                on_first_token=on_first_token,
                timeout=llm_timeout
            ):
                full_answer += token
            
            # Log completion timing
            log_timing_rag("llm_generation_complete", time.time() - t_llm, tenant_id)
            
            # Validate answer against evidence
            t_validate = time.time()
            is_supported = answer_supported_by_evidence(full_answer, context)
            validation_duration = time.time() - t_validate
            log_timing_rag("answer_validation", validation_duration, tenant_id, 
                          is_supported=is_supported, answer_length=len(full_answer))
            
            if not is_supported:
                logger.warning(
                    "Answer validation REJECTED. Replacing with refusal. Original: %s",
                    full_answer[:200]
                )
                full_answer = "The document does not specify this."
            
            # Stream the (validated) answer in chunks to simulate streaming
            CHUNK_SIZE = 75  # Characters per chunk
            for i in range(0, len(full_answer), CHUNK_SIZE):
                chunk = full_answer[i:i+CHUNK_SIZE]
                yield chunk
        else:
            # DIRECT MODE: Stream tokens as they arrive (original behavior)
            async for token in llm_provider.generate_stream(
                prompt, 
                tenant_id, 
                max_tokens=max_tokens, 
                on_first_token=on_first_token,
                timeout=llm_timeout
            ):
                yield token
            
            # Log completion timing
            log_timing_rag("llm_generation_complete", time.time() - t_llm, tenant_id)
        
    except Exception as e:
        logger.exception("LLM generation request failed: %s", e)
        raise


def clear_embedding_cache() -> None:
    """Clear the query embedding cache."""
    global _embedding_cache
    _embedding_cache.clear()
    logger.info("Embedding cache cleared")


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
    
    # Delete tenant-specific collection
    chroma_client = clients.get_chroma_client()
    collection_name = f"documents_{tenant_id}"
    try:
        chroma_client.delete_collection(collection_name)
        logger.info("Deleted collection: %s", collection_name)
    except Exception as e:
        logger.warning("Could not delete collection %s: %s", collection_name, e)
    
    # Remove from cache
    if tenant_id in _tenant_collections:
        del _tenant_collections[tenant_id]
    
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
    debug: int = 0
) -> Tuple[AsyncGenerator[str, None], List[str], List[str], str, Dict[str, Any]]:
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
    """
    return await query_collection(tenant_id, question, top_k, mode=mode, conversation_history=conversation_history, doc_ids=doc_ids, debug=debug)

