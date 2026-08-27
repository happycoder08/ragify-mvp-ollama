/**
 * TypeScript types matching RAGify backend API schemas.
 * DO NOT modify these - they must match backend exactly.
 */

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  tenant_id: string;
}

export interface QueryRequest {
  question: string;
  top_k?: number;
  mode?: 'fast' | 'full';
  conversation_id?: number;
  doc_ids?: number[];
  debug?: number; // 0=off, 1=detailed, 2=verbose
  stream?: boolean;
}

export interface EvidenceItem {
  snippet: string;
  chunk_id: string;
  heading?: string;
  doc_id?: number;
}

export interface SourceItem {
  doc_id?: number;
  filename: string;
  chunk_id?: string;
}

export interface DebugInfo {
  evidence_count: number;
  sources_count: number;
  retrieved_count?: number;
  selected_count?: number;
  request_id?: string;
  tenant_id?: string;
  collection_name?: string;
  collection_count?: number;
  doc_ids_filter?: number[];
  top10_scores?: Array<[number, string]>;
  grounding_gate?: Record<string, unknown>;
  selected_chunks?: Array<Record<string, unknown>>;
  context?: string;
  total_retrieved?: number;
  k_final?: number;
  selected_chunk_ids?: string[];
  selected_headings?: string[];
  context_text_chars?: number;
  context_chunks_count?: number;
}

export interface QueryFinalResponse {
  answer: string;
  refused: boolean;
  refusal_reason?: string;
  evidence: EvidenceItem[];
  sources: SourceItem[];
  debug_info?: DebugInfo;
}

export interface DocumentItem {
  id: number;
  filename: string;
  status: 'pending' | 'indexed' | 'failed';
  created_at: string;
  updated_at: string;
  error_message?: string;
}

export interface UploadResponse {
  status: string;
  message: string;
  documents: DocumentItem[];
  files_processed: number;
  files_with_db_record: number;
}

export interface DocumentsListResponse {
  documents: DocumentItem[];
}

// SSE Event types from backend
export interface SSETokenEvent {
  t: string; // token chunk
}

export interface SSEDebugEvent extends DebugInfo {}

export interface SSEFinalEvent extends QueryFinalResponse {}

export interface SSEErrorEvent {
  detail: string;
}
