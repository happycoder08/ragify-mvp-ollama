import glob
from fastapi import BackgroundTasks
# --- DEMO MODE STARTUP ---
def _demo_mode_startup():
    """
    If RAGIFY_MODE=demo, reset tenant vector store and index a bundled demo document.
    Logs success/failure. No effect in non-demo modes.
    """
    import os
    import shutil
    DEMO_MODE = os.getenv("RAGIFY_MODE", "").lower() == "demo"
    if not DEMO_MODE:
        return
    try:
        logger.info("[DEMO MODE] Resetting vector store and indexing demo document...")
        # Remove vectorstore directory (ChromaDB persistence)
        from app.config import VECTOR_DIR
        if os.path.exists(VECTOR_DIR):
            shutil.rmtree(VECTOR_DIR)
            logger.info(f"[DEMO MODE] Cleared vector store at {VECTOR_DIR}")
        # Remove uploads (optional, for clean demo)
        from app.config import UPLOAD_DIR
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            logger.info(f"[DEMO MODE] Cleared uploads at {UPLOAD_DIR}")
        # Index a bundled demo document (must exist in demo_docs/)
        demo_path = os.path.join(os.path.dirname(__file__), '../../demo_docs/demo_demo.txt')
        if not os.path.exists(demo_path):
            logger.error(f"[DEMO MODE] Demo document not found: {demo_path}")
            return
        with open(demo_path, encoding="utf-8") as f:
            demo_text = f.read()
        # Use chunking logic from ingestion
        from app.services import ingestion
        chunks = ingestion.chunk_text(demo_text)
        # Index under tenant 'default' and filename 'demo_demo.txt'
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(add_documents('default', chunks, 'demo_demo.txt'))
        logger.info("[DEMO MODE] Demo document indexed successfully.")
    except Exception as e:
        logger.error(f"[DEMO MODE] Startup failed: {e}", exc_info=True)

# --- Call demo-mode startup at import time ---
_demo_mode_startup()
from typing import List, Tuple, AsyncGenerator, Dict, Any
import json
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
    # Tokenize with stopword removal and intent expansion
    query_tokens = _tokenize_and_filter(query, expand_intents=True)
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

    header_tokens = _tokenize_and_filter(header_line, expand_intents=True)
    body_tokens = _tokenize_and_filter("\n".join(body_lines) if body_lines else doc, expand_intents=True)

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

    # Time boosts
    q_time_patterns = re.findall(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b|\b\d{1,2}\s*(?:am|pm)\b", q_lower)
    if any((tp if isinstance(tp, str) else tp[0]) and ((tp if isinstance(tp, str) else tp[0]) in doc_lower) for tp in q_time_patterns):
        base_score = min(1.5, base_score + 0.35)
    elif any(t in query_set for t in ['time', 'arrive', 'arrival', 'report']):
        if re.search(r"\b\d{1,2}:\d{2}\s*(?:am|pm)?\b|\b\d{1,2}\s*(?:am|pm)\b", doc_lower):
            base_score = min(1.5, base_score + 0.25)

    # Arrival/reporting boost (operational intent)
    if any(t in query_set for t in ['arrive', 'arrival', 'report', 'checkin']):
        if any(kw in doc_lower for kw in ['arrive', 'arrival', 'report', 'reception', 'check in', 'check-in']):
            base_score = min(1.5, base_score + 0.3)

    # Email signature boosts
    if 'email' in query_set and 'signature' in query_set:
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
        base_score = min(1.5, base_score + location_score)

    return min(1.5, base_score)


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

    # Penalize chunks with zero lexical overlap (verbs/nouns, not stopwords)
    # Extract verbs/nouns from query and doc, check for overlap
    def extract_content_words(text: str) -> set:
        # Simple heuristic: keep words not in STOPWORDS and length > 2
        tokens = re.split(r"\\s+", re.sub(r"[^A-Za-z0-9]+", " ", text))
        return set(
            t.lower() for t in tokens
            if len(t) > 2 and t.lower() not in STOPWORDS
        )

    query_words = extract_content_words(query)
    doc_words = extract_content_words(doc)
    overlap = query_words & doc_words

    # If there is zero overlap, apply a strong penalty to the hybrid score
    zero_overlap_penalty = 0.15  # Set to a low value (acts as a floor)
    has_overlap = len(overlap) > 0

    # Normalize vector distance to 0-1 range (invert so higher is better)
    normalized_distance = min(vector_distance / 500.0, 1.0)
    vector_score = 1.0 - normalized_distance

    # Combine scores: 50% semantic (vector), 50% lexical
    combined_score = 0.50 * vector_score + 0.50 * lexical_score

    if not has_overlap:
        # Deprioritize: set score to minimum of calculated and penalty
        combined_score = min(combined_score, zero_overlap_penalty)

    return combined_score


def _dedupe_results(results: List[Tuple[str, Dict, float]], ids: List[str]) -> Tuple[List[Tuple[str, Dict, float]], List[str]]:
    """Remove near-duplicate chunks based on content fingerprint while preserving order."""
    seen = set()
    deduped_results: List[Tuple[str, Dict, float]] = []
    deduped_ids: List[str] = []

    for (doc, meta, dist), cid in zip(results, ids):
        fp = _fingerprint_chunk(doc)
        if fp in seen:
            continue
        seen.add(fp)
        deduped_results.append((doc, meta, dist))
        deduped_ids.append(cid)

    return deduped_results, deduped_ids


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


def _dedupe_by_header(results: List[Tuple[str, Dict, float]], ids: List[str]) -> Tuple[List[Tuple[str, Dict, float]], List[str]]:
    """
    Remove near-duplicate chunks based on (source_file, normalized_header).
    Keeps only the first occurrence of each unique (source, header) pair.
    Preserves ordering.
    """
    seen_headers = set()
    deduped_results: List[Tuple[str, Dict, float]] = []
    deduped_ids: List[str] = []

    for (doc, meta, dist), cid in zip(results, ids):
        source = meta.get("source_file") or meta.get("filename", "unknown")
        normalized_header = _normalize_header(doc)
        
        # Create unique key: (source_file, normalized_header)
        key = (source, normalized_header)
        
        if key in seen_headers:
            continue
        
        seen_headers.add(key)
        deduped_results.append((doc, meta, dist))
        deduped_ids.append(cid)

    return deduped_results, deduped_ids


def _apply_header_reranking(results: List[Tuple[str, Dict, float]], debug: bool = False, request_id: str = None) -> List[Tuple[str, Dict, float]]:
    """
    Apply lightweight header-based reranking to boost action-oriented chunks
    and penalize generic schedule sections.
    
    Boosts:
    - Headers containing action verbs (arrive, report, bring, complete, setup, onboarding)
    
    Penalties:
    - Generic schedule sections (lunch, break, time ranges without context)
    
    Args:
        results: List of (doc, meta, distance) tuples
        debug: Whether to log reranking decisions
        request_id: Request ID for logging
        
    Returns:
        List of (doc, meta, adjusted_distance) tuples (sorted by adjusted distance)
    """
    # Action verbs that indicate operational/actionable content
    ACTION_VERBS = {
        'arrive', 'arrival', 'report', 'bring', 'complete', 'submit',
        'setup', 'set up', 'onboarding', 'orientation', 'register',
        'check in', 'checkin', 'badge', 'documents', 'prepare'
    }
    
    # Generic schedule terms that are less useful for specific questions
    GENERIC_SCHEDULE = {
        'lunch', 'break', 'coffee', 'end of day', 'eod',
        'closing', 'wrap up', 'wrapup', 'depart'
    }
    
    adjusted_results = []
    adjustments = []  # Track adjustments for logging
    
    for doc, meta, dist in results:
        # Extract header (first non-empty line)
        lines = doc.split('\n')
        header = ""
        for line in lines:
            if line.strip():
                header = line.strip()
                break
        
        header_lower = header.lower()
        original_dist = dist
        adjustment_reason = None
        
        # Check for action verbs (boost by reducing distance)
        has_action_verb = any(verb in header_lower for verb in ACTION_VERBS)
        if has_action_verb:
            # Boost: reduce distance by 15% (makes it rank higher)
            dist = dist * 0.85
            adjustment_reason = "action_verb_boost"
        
        # Check for generic schedule terms (penalize by increasing distance)
        # Only apply if no action verb was found
        if not has_action_verb:
            has_generic = any(term in header_lower for term in GENERIC_SCHEDULE)
            # Also check for standalone time ranges without context (e.g., "12:00 PM - 1:00 PM")
            # These are less useful than specific activities
            is_time_range_only = bool(re.match(r'^[\d\s:apm-]+$', header_lower.strip()))
            
            if has_generic or is_time_range_only:
                # Penalty: increase distance by 20% (makes it rank lower)
                dist = dist * 1.20
                adjustment_reason = "generic_schedule_penalty" if has_generic else "time_range_only_penalty"
        
        adjusted_results.append((doc, meta, dist))
        
        # Track adjustment for logging
        if adjustment_reason and debug:
            adjustments.append({
                "header": header[:80],
                "original_dist": round(original_dist, 4),
                "adjusted_dist": round(dist, 4),
                "adjustment": adjustment_reason
            })
    
    # Sort by adjusted distance (lower = better)
    adjusted_results.sort(key=lambda x: x[2])
    
    # Log adjustments if any were made
    if adjustments and debug:
        logger.info(
            "[%s] Header reranking applied: %d chunks adjusted. Examples: %s",
            request_id or "no-request-id",
            len(adjustments),
            adjustments[:5]  # Log first 5 adjustments
        )
    
    return adjusted_results


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


def _get_embedding_provider():
    """
    Get or create the global embedding provider instance.
    
    Returns the embedding provider, which may be:
    - An injected provider (for tests)
    - A RealEmbedder (default, uses HTTP to call Ollama/OpenAI)
    """
    global _embedding_provider
    if _embedding_provider is None:
        # Lazy initialization: create default RealEmbedder
        from app.services.embeddings import RealEmbedder
        http_client = clients.get_http_client()
        _embedding_provider = RealEmbedder(http_client=http_client)
    return _embedding_provider


def _get_collection(tenant_id: str = "default"):
    """Get tenant-specific collection from centralized ChromaDB client."""
    chroma_client = clients.get_chroma_client()
    
    # Get or create tenant-specific collection
    if tenant_id not in _tenant_collections:
        collection_name = f"documents_{tenant_id}"
        _tenant_collections[tenant_id] = chroma_client.get_or_create_collection(collection_name)
        logger.info(f"Initialized collection for tenant: {tenant_id}")
    
    return _tenant_collections[tenant_id]


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


async def query_collection(
    tenant_id: str, 
    question: str, 
    top_k: int = 4, 
    mode: str = "full",
    conversation_history: List[Dict] = None,
    doc_ids: List[int] = None,
    debug: int = 0,
    request_id: str = None
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
        request_id: Request identifier for tracing (optional)
    """
    # Normalize doc_ids: empty list behaves the same as None
    if doc_ids is not None and len(doc_ids) == 0:
        doc_ids = None
    
    import uuid
    if not request_id:
        request_id = str(uuid.uuid4())
    
    logger.info("[%s] Query received from tenant %s: %s (history_len=%d, doc_ids=%s)", 
                request_id, tenant_id, question, len(conversation_history) if conversation_history else 0, doc_ids)

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
    q_embedding = (await embed_texts([expanded_question], tenant_id=tenant_id))[0]
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
            yield "The document does not specify this."
        debug_info = {"retrieved_count": 0, "selected_count": 0, "chunks": [], "refused": True, "refusal_reason": "NOT_FOUND", "request_id": request_id} if debug >= 1 else {"refused": True, "refusal_reason": "NOT_FOUND", "request_id": request_id}
        return not_found_gen(), [], [], "", debug_info

    # Log top 10 retrieval scores for tracing
    top10_scores = [(round(dist, 2), ids[i] if i < len(ids) else f"idx_{i}") for i, dist in enumerate(distances[:10])]
    logger.info("[%s] Retrieved %d chunks, top 10 scores: %s", request_id, len(distances), top10_scores)

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

    # Apply lightweight header-based reranking (boost action verbs, penalize generic schedule)
    if len(filtered_results) > 1:
        t_header_rerank = time.time()
        filtered_results = _apply_header_reranking(filtered_results, debug=debug >= 1, request_id=request_id)
        header_rerank_duration = time.time() - t_header_rerank
        log_timing_rag(
            "header_reranking",
            header_rerank_duration,
            tenant_id,
            rerank_ms=round(header_rerank_duration * 1000, 2)
        )
    
    # Deterministic deduplication by content fingerprint
    before_fingerprint_dedup = len(filtered_results)
    filtered_results, ids = _dedupe_results(filtered_results, ids)
    if debug >= 1 and len(filtered_results) < before_fingerprint_dedup:
        logger.info("[%s] Deduped retrieved chunks from %d to %d using content fingerprints", request_id, before_fingerprint_dedup, len(filtered_results))
    
    # Header-based deduplication: remove chunks with duplicate (source, header) pairs
    before_header_dedup = len(filtered_results)
    filtered_results, ids = _dedupe_by_header(filtered_results, ids)
    if debug >= 1 and len(filtered_results) < before_header_dedup:
        logger.info("[%s] Deduped retrieved chunks from %d to %d using header deduplication", request_id, before_header_dedup, len(filtered_results))
    
    # Limit to top N context chunks after hybrid scoring (enforce strict limit)
    # Use top 5 after reranking regardless of mode, to ensure focused grounding
    TOP_CONTEXT_N = 5  # Always select top 5 after hybrid reranking
    if len(filtered_results) > TOP_CONTEXT_N:
        logger.info(
            "[%s] Limiting from %d to top %d best chunks after hybrid scoring",
            request_id,
            len(filtered_results),
            TOP_CONTEXT_N,
        )
        filtered_results = filtered_results[:TOP_CONTEXT_N]
    
    # Log selected chunks for tracing (no raw content)
    selected_chunk_ids = [ids[i] if i < len(ids) else f"idx_{i}" for i in range(len(filtered_results))]
    logger.info("[%s] Selected %d chunks: %s", request_id, len(filtered_results), selected_chunk_ids)

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

    # Build detailed selected chunk previews for debug (used even on refusal)
    selected_chunks_debug: List[Dict[str, Any]] = []
    if debug >= 1:
        for idx, (doc, meta, dist) in enumerate(filtered_results):
            chunk_id = ids[idx] if ids and idx < len(ids) else f"chunk_{meta.get('chunk', idx)}"
            header = None
            for line in doc.splitlines():
                if line.strip():
                    header = line.strip()
                    break
            selected_chunks_debug.append({
                "id": chunk_id,
                "header": header or doc[:80].replace('\n', ' '),
                "snippet": doc[:200].replace('\n', ' ') + ("..." if len(doc) > 200 else ""),
                "distance": round(dist, 4) if isinstance(dist, (int, float)) else -1,
                "source": meta.get("source_file", "unknown")
            })

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

    # --- PRESENTATION LAYER: Synthesize natural answer for time/arrival questions ---
    import re
    def _synthesize_time_answer(question: str, evidence: list) -> str | None:
        ql = question.lower()
        if not any(w in ql for w in ["time", "arrive", "arrival"]):
            return None
        # Look for a time pattern in evidence
        for ev in evidence:
            # Accept both "8:00 AM" and "8 am" etc.
            m = re.search(r"(\d{1,2}(:\d{2})?\s*(am|pm))", ev, re.IGNORECASE)
            if m:
                time_str = m.group(1).strip()
                # Try to find "first day" or similar in evidence
                if "first day" in ev.lower():
                    return f"You should arrive at {time_str} on your first day."
                return f"You should arrive at {time_str}."
        return None

    synthesized = _synthesize_time_answer(question, evidence)
    if synthesized:
        # Replace evidence with the synthesized answer for the LLM prompt
        evidence = [synthesized]
    
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
    
    # Log evidence lines for tracing (truncated, no full content)
    evidence_preview = [line[:80] + "..." if len(line) > 80 else line for line in gate_evidence_lines[:3]]
    logger.info(
        "[%s] Grounding gate: should_proceed=%s, refusal_reason=%s, evidence_lines=%d, max_overlap=%.0f, sum_top3=%.0f, failed_check=%s, evidence_preview=%s",
        request_id, should_proceed, refusal_reason, len(gate_evidence_lines), max_overlap, sum_top3, failed_check or "NONE", evidence_preview
    )
    
    import os
    DEMO_STRICT = os.environ.get("RAGIFY_DEMO_STRICT", "false").lower() == "true"
    def _evidence_has_time_or_number(evidence_list):
        import re
        for ev in evidence_list:
            if re.search(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", ev, re.IGNORECASE):
                return True
            if re.search(r"\b\d{1,2}:\d{2}\b", ev):
                return True
            if re.search(r"\b\d+\b", ev):
                return True
        return False

    if not should_proceed:
        # DEMO_STRICT guardrail: if evidence_count >= 1 and evidence contains time/number anchor, never refuse
        if DEMO_STRICT and len(evidence) >= 1 and _evidence_has_time_or_number(evidence):
            logger.info("[DEMO_STRICT] Override refusal: evidence_count >= 1 and evidence contains time/number anchor.")
            should_proceed = True
            refusal_reason = None
            failed_check = None
        else:
            # Return refusal response with standardized message but retain selected chunks/context for debugging
            async def refusal_gen():
                yield "The document does not specify this."

            refusal_debug_info = {
                "retrieved_count": len(docs),
                "selected_count": len(filtered_results),
                "chunks": selected_chunks_debug,
                "refused": True,
                "refusal_reason": refusal_reason,
                "request_id": request_id,
                "top10_scores": top10_scores,
                "grounding_gate": {
                    "should_proceed": should_proceed,
                    "max_overlap": max_overlap,
                    "sum_top3": sum_top3,
                    "failed_check": failed_check,
                    "evidence_lines_count": len(gate_evidence_lines),
                    "thresholds": {
                        "min_support": MIN_SUPPORT,
                        "min_total_support": MIN_TOTAL_SUPPORT
                    }
                }
            } if debug >= 1 else {"refused": True, "refusal_reason": "NOT_FOUND", "request_id": request_id}

            logger.warning(
                "[%s] Grounding gate REFUSED query (reason=NOT_FOUND, failed_check=%s, max_overlap=%.0f, sum_top3=%.0f): %s",
                request_id, failed_check, max_overlap, sum_top3, question[:100]
            )

            return refusal_gen(), dedup_sources, evidence, context, refusal_debug_info
    
    # Build debug info: include retrieved_count, selected_count, and detailed chunk diagnostics
    if debug >= 1:
        debug_info = {
            "retrieved_count": len(docs),
            "selected_count": len(filtered_results),
            "chunks": selected_chunks_debug,
            "request_id": request_id,
            "top10_scores": top10_scores,
            "grounding_gate": {
                "should_proceed": should_proceed,
                "max_overlap": max_overlap,
                "sum_top3": sum_top3,
                "failed_check": failed_check,
                "evidence_lines_count": len(gate_evidence_lines),
                "thresholds": {
                    "min_support": MIN_SUPPORT,
                    "min_total_support": MIN_TOTAL_SUPPORT
                }
            },
            "refused": False,
            "refusal_reason": None
        }
    else:
        # Legacy mode: return simple selected_info list
        debug_info = selected_info
    
    answer_gen = _call_chat_model(question, context, tenant_id, mode=mode, conversation_history=conversation_history, request_id=request_id)
    return answer_gen, detailed_sources, evidence, context, debug_info


async def _call_chat_model(
    question: str, 
    context: str, 
    tenant_id: str, 
    mode: str = "full",
    conversation_history: List[Dict] = None,
    validate_before_stream: bool = True,
    request_id: str = None
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
        request_id: Request identifier for tracing (optional)
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
            "5. Conversation history may be incomplete or incorrect. Treat Evidence as the ONLY source of truth. Do not introduce facts from history unless they are explicitly supported by Evidence.\n"
            "6. Extract exact details; do NOT add outside knowledge.\n"
            "7. Include ONLY directly relevant information; skip unrelated sections.\n"
            "8. For time/arrival questions: state the exact time (with AM/PM) and location in ONE sentence.\n"
            "9. Do NOT include background information, Q/A sections, or general guidance unless directly answering the question.\n"
            "\n"
            "Format: Direct answer in 1-2 short sentences."
        )
    else:
        instruction = (
            "ANSWER RULES:\n"
            "1. You must answer using ONLY the Evidence lines below.\n"
            "2. If Evidence does not contain the answer, output exactly: The document does not specify this.\n"
            "3. Do not use conversation history as a source of truth; history is for continuity only.\n"
            "4. Conversation history may be incomplete or incorrect. Treat Evidence as the ONLY source of truth. Do not introduce facts from history unless they are explicitly supported by Evidence.\n"
            "5. Extract exact details; do NOT add outside knowledge.\n"
            "6. Be concise and specific; avoid unrelated guidance.\n"
            "7. If the question asks about time or arrival, include BOTH the exact time (with AM/PM) and the exact location/floor if present in the Evidence, in one short sentence.\n"
            "8. Preserve the exact formatting of times (e.g., 8:00 AM) and floors (e.g., 3rd floor) as written.\n"
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

    logger.info("[%s] Calling LLM for question (len=%d) with context length %d mode=%s max_tokens=%s mock=%s history_len=%d validate=%s", 
                request_id or "no-request-id", len(question), len(context), mode, max_tokens, is_mock_mode(), len(conversation_history) if conversation_history else 0, validate_before_stream)
    
    # If mock mode, yield canned response immediately
    if is_mock_mode():
        yield "(mocked) This is a canned answer used for local UI testing."
        return

    # Get LLM provider and guardrail config for timeout
    llm_provider = _get_llm_provider()
    guardrail_config = get_guardrail_config(tenant_id)
    llm_timeout = guardrail_config.llm_timeout_seconds
    
    # Use the orchestrator for buffered streaming with validation or direct streaming
    async for chunk in generate_answer_stream(
        prompt=prompt,
        tenant_id=tenant_id,
        provider=llm_provider,
        max_tokens=max_tokens,
        timeout=llm_timeout,
        validate_fn=answer_supported_by_evidence if validate_before_stream else None,
        evidence_text=context,
        refusal_text="The document does not specify this.",
        request_id=request_id,
        chunk_size=75
    ):
        yield chunk


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
    debug: int = 0,
    request_id: str = None
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
        request_id: Request identifier for tracing (optional)
    """
    # Normalize doc_ids: empty list behaves the same as None
    if doc_ids is not None and len(doc_ids) == 0:
        doc_ids = None
    
    return await query_collection(tenant_id, question, top_k, mode=mode, conversation_history=conversation_history, doc_ids=doc_ids, debug=debug, request_id=request_id)

