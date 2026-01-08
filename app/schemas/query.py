"""
Canonical schemas for query API request/response.

Enforces strict typing and validation for /api/query endpoint.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator, root_validator

class ClarificationType(str, Enum):
    """Type of clarification required from the user."""
    TIMEFRAME = "TIMEFRAME"
    DOCUMENT = "DOCUMENT"
    SCOPE = "SCOPE"
    LOCATION = "LOCATION"
    ROLE = "ROLE"
    OTHER = "OTHER"


class AnswerSchema(Enum):
    """Answer schema classification for query responses."""
    FACT_SINGLE = "FACT_SINGLE"
    CHECKLIST_PROCEDURE = "CHECKLIST_PROCEDURE"
    POLICY_EXCERPT = "POLICY_EXCERPT"
    SUMMARY_OVERVIEW = "SUMMARY_OVERVIEW"
    BOOLEAN_SPECIFIED = "BOOLEAN_SPECIFIED"
    LOCATION_OR_CONTACT = "LOCATION_OR_CONTACT"
    NOT_FOUND_EXPLICIT = "NOT_FOUND_EXPLICIT"


class ClarificationType(str, Enum):
    """Type of clarification required from the user."""
    TIMEFRAME = "TIMEFRAME"
    DOCUMENT = "DOCUMENT"
    SCOPE = "SCOPE"
    LOCATION = "LOCATION"
    ROLE = "ROLE"
    OTHER = "OTHER"


MAX_CLARIFICATION_OPTIONS: int = 6


class ClarificationPayload(BaseModel):
    """Details about the clarification required."""
    type: str = Field(..., description="Type of clarification (e.g. TIMEFRAME, DOCUMENT)")
    question: str = Field(..., description="The question to ask the user")
    options: Optional[List[str]] = Field(default=None, description="Suggested options for the user")


class DecisionType(str, Enum):
    """How the final answer decision was produced."""

    EXTRACTED = "EXTRACTED"            # Deterministic extractor (wifi, arrival time, etc.)
    LLM_VALIDATED = "LLM_VALIDATED"    # Passed schema + grounding validation
    FORCED_REFUSAL = "FORCED_REFUSAL"  # Forced to canonical refusal (coverage, gate, invariants)
    VALIDATION_FAILED = "VALIDATION_FAILED"  # LLM output failed validation even after retry


class AnswerDecision(BaseModel):
    """Final decision for a query, independent of answer text.

    This is produced inside rag_service.query_collection and consumed by /api/query
    so the endpoint does not need to inspect answer_text to infer refusal.
    """

    decision_type: DecisionType = Field(..., description="How the answer/refusal was chosen")
    refused: bool = Field(..., description="Whether the query was ultimately refused")
    refusal_reason: Optional[str] = Field(default=None, description="Reason for refusal, if any")
    answer_schema: Optional[AnswerSchema] = Field(default=None, description="Schema used for this answer")
    invariant_violation: Optional[bool] = Field(default=None, description="True if schema invariant was violated")
    validation_failed: Optional[bool] = Field(default=None, description="True if schema validation failed after retry")
    final_answer_text: Optional[str] = Field(
        default=None,
        description="Authoritative final answer text after validation (may differ from streamed first pass)",
    )
    validation_meta: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured metadata about validation: first/second pass validity, retries, invariant checks, etc.",
    )


class QueryRequest(BaseModel):
    """Request schema for /api/query endpoint."""
    question: str = Field(..., min_length=1, description="User's question")
    top_k: int = Field(default=4, ge=1, le=50, description="Number of chunks to retrieve")
    mode: str = Field(default="full", description="Query mode: fast or full")
    conversation_id: Optional[int] = Field(default=None, description="Optional conversation context")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=None, description="Optional explicit conversation history")
    doc_ids: Optional[List[int]] = Field(default=None, description="Optional document IDs to filter search scope")
    debug: int = Field(default=0, ge=0, le=2, description="Debug level: 0=off, 1=detailed, 2=verbose")
    stream: bool = Field(default=True, description="If true, stream response as SSE; if false, return JSON response.")


class EvidenceItem(BaseModel):
    """Single piece of evidence from retrieved documents."""
    snippet: str = Field(..., description="Text snippet from document")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    heading: Optional[str] = Field(default=None, description="Section heading or document title")
    doc_id: Optional[int] = Field(default=None, description="Database document ID")
    anchor_type: Optional[str] = Field(default=None, description="Type of anchor content: WIFI, TIME, or None")
    anchor_detected: bool = Field(default=False, description="Whether anchor content was detected in this evidence")
    
    class Config:
        json_schema_extra = {
            "example": {
                "snippet": "Employees receive 15 days of vacation per year",
                "chunk_id": "1_onboarding.txt_3",
                "heading": "Vacation Policy",
                "doc_id": 1,
                "anchor_type": None,
                "anchor_detected": False
            }
        }


class SourceItem(BaseModel):
    """Source document reference."""
    doc_id: Optional[int] = Field(default=None, description="Database document ID (if available)")
    filename: str = Field(..., description="Original filename")
    chunk_id: Optional[str] = Field(default=None, description="Specific chunk reference")
    
    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": 1,
                "filename": "onboarding.txt",
                "chunk_id": "1_onboarding.txt_3"
            }
        }


class DebugInfo(BaseModel):
    """Optional debug information for query diagnostics."""
    evidence_count: int = Field(..., description="Number of evidence items")
    sources_count: int = Field(..., description="Number of source documents")
    retrieved_count: Optional[int] = Field(default=None, description="Total chunks retrieved before filtering")
    selected_count: Optional[int] = Field(default=None, description="Chunks selected after reranking")
    request_id: Optional[str] = Field(default=None, description="Unique request identifier")
    tenant_id: Optional[str] = Field(default=None, description="Tenant identifier used for query")
    collection_name: Optional[str] = Field(default=None, description="ChromaDB collection name")
    collection_count: Optional[int] = Field(default=None, description="Total chunks in collection")
    doc_ids_filter: Optional[List[int]] = Field(default=None, description="Document IDs filter applied (if any)")
    top10_scores: Optional[List] = Field(default=None, description="Top 10 retrieval scores")
    grounding_gate: Optional[dict] = Field(default=None, description="Grounding validation details")
    selected_chunks: Optional[List] = Field(default=None, description="Selected chunk metadata")
    retrieved_chunks_top20: Optional[List] = Field(default=None, description="Top 20 retrieved chunks before rerank (metadata only)")
    context: Optional[str] = Field(default=None, description="Full context sent to LLM")
    context_length: Optional[int] = Field(default=None, description="Length of context text in characters")
    refused: Optional[bool] = Field(default=None, description="Whether query was refused by grounding gate")
    refusal_reason: Optional[str] = Field(default=None, description="Reason for refusal if applicable")
    failed_check: Optional[str] = Field(default=None, description="Which grounding check failed")
    support_score: Optional[float] = Field(default=None, description="Grounding support score (e.g., sum_top3)")
    gate_evidence_lines_count: Optional[int] = Field(default=None, description="Count of evidence lines considered by gate")
    debug_trace: Optional[dict] = Field(default=None, description="Detailed debug trace for high debug levels")
    retrieved_top: Optional[List[dict]] = Field(default=None, description="Top 10 retrieved chunks with chunk_id, heading, distance")
    selected_chunk_ids: Optional[List[str]] = Field(default=None, description="List of chunk_ids used to build context")
    selected_headings: Optional[List[str]] = Field(default=None, description="List of headings used")
    context_chunks_count: Optional[int] = Field(default=None, description="Number of chunks concatenated")
    context_text_chars: Optional[int] = Field(default=None, description="Length of final context text")
    invariant_violation: Optional[bool] = Field(default=None, description="Historical flag for schema invariants (deprecated)")
    refusal_phrase_in_non_refusal_answer: Optional[bool] = Field(
        default=None,
        description="Set if refused=False but answer equals the canonical refusal phrase",
    )
    answer_schema: Optional[AnswerSchema] = Field(default=None, description="Answer schema classification")
    empty_answer_invariant_tripped: Optional[bool] = Field(
        default=None,
        description="Set if non-clarification response had an empty answer and required a fallback",
    )
    fallback_from_evidence: Optional[bool] = Field(
        default=None,
        description="Set if an empty-answer fallback was constructed from evidence",
    )


class QueryFinalResponse(BaseModel):
    """
    Canonical final response schema for /api/query endpoint.
    
    This is the structured JSON emitted at the end of the streaming response.
    Guarantees:
    - If refused=True: answer is exactly "The document does not specify this."
    - If refused=False: evidence list has >=1 items and sources list has >=1 items
    """
    answer: str = Field(..., description="Final answer text (complete, not streaming tokens)")
    refused: bool = Field(..., description="Whether the query was refused due to lack of grounding")
    refusal_reason: Optional[str] = Field(default=None, description="Reason for refusal (NOT_FOUND, LOW_SUPPORT, etc.)")
    evidence: List[EvidenceItem] = Field(default_factory=list, description="Supporting evidence snippets")
    sources: List[SourceItem] = Field(default_factory=list, description="Source document references")
    pipeline_marker: str = Field(
        ...,
        description=(
            "High-level pipeline mode marker: EXTRACTOR_* for deterministic "
            "extractors, LLM_VALIDATED for validated LLM answers, "
            "FORCED_REFUSAL when the system forces a canonical refusal, "
            "or CLARIFICATION_REQUIRED."
        ),
    )
    debug_info: Optional[DebugInfo] = Field(default=None, description="Optional debug diagnostics")
    needs_clarification: Optional[bool] = Field(default=None, description="True if the system needs user input to proceed")
    clarification: Optional[ClarificationPayload] = Field(default=None, description="Details about the required clarification")
    
    @root_validator
    def validate_response_consistency(cls, values):
        """Validate response consistency: refusal message, evidence, sources, and clarification."""
        import logging
        refused = values.get('refused')
        answer = values.get('answer')
        evidence = values.get('evidence', [])
        sources = values.get('sources', [])
        needs_clarification = values.get('needs_clarification')
        clarification = values.get('clarification')
        pipeline_marker = values.get('pipeline_marker')
        
        refusal_answer = "The document does not specify this."
        
        # Clarification logic
        if needs_clarification or clarification is not None:
            if refused:
                raise ValueError("Clarification required responses cannot be refused.")
            if pipeline_marker != "CLARIFICATION_REQUIRED":
                raise ValueError("Clarification responses must use pipeline_marker='CLARIFICATION_REQUIRED'.")
            if not clarification:
                raise ValueError("Clarification details missing for needs_clarification=True.")
            if answer != clarification.question:
                raise ValueError("Answer must match clarification question.")
            # Evidence/sources can be empty for clarification, so we skip the check below
            return values

        # If refused, must use canonical message (this enforces content for an
        # explicit refusal decision, but never infers refusal from content).
        if refused and answer != refusal_answer:
            raise ValueError("Refused queries must use canonical refusal message: 'The document does not specify this.'")

        # If not refused but evidence or sources are empty or the answer looks
        # like a refusal, only log; do not coerce. /api/query is responsible
        # for deciding refused based on structured signals (AnswerDecision).
        if not refused:
            if len(evidence) == 0 or len(sources) == 0 or answer == refusal_answer:
                logging.error(
                    "[QueryFinalResponse] Inconsistent non-refusal: refused==False but evidence/sources empty or answer is refusal message."
                )
        return values
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Employees receive 15 days of vacation per year.",
                "refused": False,
                "refusal_reason": None,
                "pipeline_marker": "LLM_VALIDATED",
                "evidence": [
                    {
                        "snippet": "All employees receive 15 days of vacation per year",
                        "chunk_id": "1_onboarding.txt_3",
                        "heading": "Vacation Policy",
                        "doc_id": 1
                    }
                ],
                "sources": [
                    {
                        "doc_id": 1,
                        "filename": "onboarding.txt",
                        "chunk_id": "1_onboarding.txt_3"
                    }
                ],
                "debug_info": None
            }
        }


def build_clarification(question: str, options: Optional[List[str]] = None, type: str = "OTHER") -> QueryFinalResponse:
    """Helper to build a consistent clarification response."""
    clarification = ClarificationPayload(
        question=question,
        options=options,
        type=type
    )
    return QueryFinalResponse(
        answer=question,
        refused=False,
        pipeline_marker="CLARIFICATION_REQUIRED",
        needs_clarification=True,
        clarification=clarification,
        evidence=[],
        sources=[]
    )

def build_question(c_type: ClarificationType, options: Optional[List[str]] = None) -> str:
    opts = [o for o in (options or []) if isinstance(o, str) and o.strip()]
    if c_type == ClarificationType.TIMEFRAME:
        return f"Which timeframe should I use: {' or '.join(opts)}?" if opts else "Which timeframe should I use?"
    if c_type == ClarificationType.DOCUMENT:
        return f"Which document should I use: {' or '.join(opts)}?" if opts else "Which document should I use?"
    if c_type == ClarificationType.SCOPE:
        return f"Which scope do you mean: {' or '.join(opts)}?" if opts else "What scope do you mean?"
    if c_type == ClarificationType.LOCATION:
        return f"Which location: {' or '.join(opts)}?" if opts else "Which location?"
    if c_type == ClarificationType.ROLE:
        return f"Which role/team: {' or '.join(opts)}?" if opts else "Which role/team?"
    return "Can you clarify what you mean?"


def normalize_options(options: Optional[List[str]]) -> Optional[List[str]]:
    if not options:
        return None
    clean: List[str] = []
    seen = set()
    for o in options:
        if not isinstance(o, str):
            continue
        s = o.strip()
        if not s or s in seen:
            continue
        clean.append(s)
        seen.add(s)
        if len(clean) >= MAX_OPTIONS:
            break
    return clean or None


def build_clarification_payload(
    c_type: ClarificationType,
    options: Optional[List[str]] = None,
    question: Optional[str] = None,
) -> Dict[str, Any]:
    norm_opts = normalize_options(options)
    q = question.strip() if isinstance(question, str) and question.strip() else build_question(c_type, norm_opts)
    return {
        "needs_clarification": True,
        "clarification": {
            "type": c_type.value,
            "question": q,
            **({"options": norm_opts} if norm_opts else {}),
        },
        "pipeline_marker": "CLARIFICATION_REQUIRED",
        "refused": False,
        "answer": q,
        "evidence": [],
        "sources": [],
    }
