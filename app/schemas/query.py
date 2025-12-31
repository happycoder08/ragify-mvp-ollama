"""
Canonical schemas for query API request/response.

Enforces strict typing and validation for /api/query endpoint.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator, root_validator


class QueryRequest(BaseModel):
    """Request schema for /api/query endpoint."""
    question: str = Field(..., min_length=1, description="User's question")
    top_k: int = Field(default=4, ge=1, le=50, description="Number of chunks to retrieve")
    mode: str = Field(default="full", description="Query mode: fast or full")
    conversation_id: Optional[int] = Field(default=None, description="Optional conversation context")
    doc_ids: Optional[List[int]] = Field(default=None, description="Optional document IDs to filter search scope")
    debug: int = Field(default=0, ge=0, le=2, description="Debug level: 0=off, 1=detailed, 2=verbose")
    stream: bool = Field(default=True, description="If true, stream response as SSE; if false, return JSON response.")


class EvidenceItem(BaseModel):
    """Single piece of evidence from retrieved documents."""
    snippet: str = Field(..., description="Text snippet from document")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    heading: Optional[str] = Field(default=None, description="Section heading or document title")
    doc_id: Optional[int] = Field(default=None, description="Database document ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "snippet": "Employees receive 15 days of vacation per year",
                "chunk_id": "1_onboarding.txt_3",
                "heading": "Vacation Policy",
                "doc_id": 1
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
    debug_info: Optional[DebugInfo] = Field(default=None, description="Optional debug diagnostics")
    
    @root_validator
    def validate_response_consistency(cls, values):
        """Validate response consistency: refusal message, evidence, and sources."""
        import logging
        refused = values.get('refused')
        answer = values.get('answer')
        evidence = values.get('evidence', [])
        sources = values.get('sources', [])
        refusal_answer = "The document does not specify this."

        # If answer is the canonical refusal message, refused must be True
        if answer == refusal_answer and not refused:
            logging.warning("[QueryFinalResponse] Coercing to refusal: answer is refusal message but refused==False.")
            values['refused'] = True
            values['evidence'] = []
            values['sources'] = []
            return values

        # If refused, must use canonical message
        if refused and answer != refusal_answer:
            raise ValueError("Refused queries must use canonical refusal message: 'The document does not specify this.'")

        # If not refused, must have evidence and sources and not be the refusal message
        if not refused:
            if len(evidence) == 0 or len(sources) == 0 or answer == refusal_answer:
                logging.warning("[QueryFinalResponse] Coercing to refusal: refused==False but evidence/sources empty or answer is refusal message.")
                values['refused'] = True
                values['answer'] = refusal_answer
                values['evidence'] = []
                values['sources'] = []
        return values
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Employees receive 15 days of vacation per year.",
                "refused": False,
                "refusal_reason": None,
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
