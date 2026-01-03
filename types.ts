/**
 * TypeScript types for RAGify API
 * Generated from FastAPI backend schemas
 */

// ============================================================================
// Authentication
// ============================================================================

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  tenant_id: string;
}

// ============================================================================
// Query (RAG Search)
// ============================================================================

export interface QueryRequest {
  question: string;
  top_k?: number;
  mode?: "fast" | "full";
  conversation_id?: number;
  doc_ids?: number[];
  debug?: number; // 0 = no debug, 1 = basic, 2 = verbose
}

export interface EvidenceItem {
  snippet: string;
  chunk_id: string;
  heading?: string | null;
  doc_id?: number | null;
}

export interface SourceItem {
  doc_id?: number | null;
  filename: string;
  chunk_id?: string | null;
}

export interface DebugInfo {
  evidence_count: number;
  sources_count: number;
  retrieved_count?: number | null;
  selected_count?: number | null;
  request_id?: string | null;
  tenant_id?: string | null;
  collection_name?: string | null;
  collection_count?: number | null;
  doc_ids_filter?: number[] | null;
  top10_scores?: number[] | null;
  grounding_gate?: Record<string, any> | null;
  selected_chunks?: any[] | null;
  context?: string | null;
  retrieved_top?: Array<{chunk_id: string, heading: string, distance: number}> | null;
  selected_chunk_ids?: string[] | null;
  selected_headings?: string[] | null;
  context_chunks_count?: number | null;
  context_text_chars?: number | null;
  invariant_violation?: boolean | null;
}

export interface QueryFinalResponse {
  answer: string;
  refused: boolean;
  refusal_reason?: string | null;
  evidence: EvidenceItem[];
  sources: SourceItem[];
  debug_info?: DebugInfo | null;
}

// SSE streaming events
export type SSEEvent =
  | { event: "debug"; data: DebugInfo }
  | { event: "token"; data: { t: string } }
  | { event: "final"; data: QueryFinalResponse };

// ============================================================================
// Documents
// ============================================================================

export interface DocumentRecord {
  id: number;
  filename: string;
  status: "pending" | "indexed" | "failed";
  created_at: string; // ISO datetime
  updated_at: string; // ISO datetime
  error_message?: string | null;
}

export interface UploadResponse {
  status: "ok";
  message: string;
  documents: DocumentRecord[];
  files_processed: number;
  files_with_db_record: number;
}

export interface DocumentsListResponse {
  documents: DocumentRecord[];
}

export interface DocumentStatusResponse extends DocumentRecord {}

export interface ReindexResponse {
  status: "ok";
  message: string;
  document: DocumentRecord;
}

export interface PurgeResponse {
  status: "ok";
  deleted: number;
  removed_files: number;
  message: string;
}

// ============================================================================
// Conversations
// ============================================================================

export interface ConversationCreate {
  title?: string;
}

export interface MessageCreate {
  role: "user" | "assistant";
  content: string;
  sources?: string[];
}

export interface Message {
  id: number;
  conversation_id: number;
  role: "user" | "assistant";
  content: string;
  sources?: string | null; // JSON string
  created_at: string; // ISO datetime
}

export interface Conversation {
  id: number;
  tenant_id: string;
  title: string;
  created_at: string; // ISO datetime
  updated_at: string; // ISO datetime
  messages?: Message[]; // Only included when fetching single conversation
}

export interface ConversationListResponse extends Array<Conversation> {}

export interface DeleteConversationResponse {
  status: "ok";
  deleted_id: number;
}

// ============================================================================
// Configuration & Health
// ============================================================================

export interface HealthResponse {
  status: "ok" | "degraded";
  mock_mode: boolean;
  ragify_mode: string;
  error?: string;
}

export interface SystemConfigResponse {
  ragify_mode: string;
  default_mode: string;
  top_k_fast: number;
  top_k_full: number;
  enable_timing_logs: boolean;
  max_conversation_turns: number;
  [key: string]: any; // Additional config fields
}

export interface GuardrailConfig {
  max_file_size_mb: number;
  max_files_per_upload: number;
  allowed_extensions: string[];
  rate_limit_requests_per_hour: number;
  rate_limit_mb_per_hour: number;
}

export interface RateLimitStatus {
  requests_count: number;
  requests_limit: number;
  upload_mb: number;
  upload_limit_mb: number;
  window_start: string; // ISO datetime
  window_end: string; // ISO datetime
}

export interface DependencyHealthResponse {
  status: "ok" | "error";
  ollama_ok: boolean;
  ollama_models: string[];
  chroma_ok: boolean;
  chroma_count: number;
}

// ============================================================================
// Debug Endpoints
// ============================================================================

export interface ChunkMatch {
  chunk_id: string;
  source: string;
  header: string;
  preview: string;
}

export interface FindChunksResponse {
  status: "ok";
  tenant_id: string;
  count: number;
  chunks: ChunkMatch[];
}

// ============================================================================
// Generic Responses
// ============================================================================

export interface ErrorResponse {
  detail: string;
}

export interface ResetResponse {
  status: "ok";
  message: string;
}

// ============================================================================
// API Client Configuration
// ============================================================================

export interface ApiConfig {
  baseUrl: string;
  token?: string;
}

// Helper type for authentication headers
export interface AuthHeaders {
  Authorization: string;
}
