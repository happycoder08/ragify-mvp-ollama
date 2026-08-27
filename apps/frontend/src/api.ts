/**
 * API client for RAGify backend
 * Uses exact contract types from src/contracts/types.ts
 * Base URL from VITE_API_URL environment variable
 */

import type {
  LoginRequest,
  LoginResponse,
  DocumentsListResponse,
  UploadResponse,
  ErrorResponse,
  PurgeResponse,
  ConversationCreate,
  Conversation,
  ConversationListResponse,
  MessageCreate,
  Message,
  DeleteConversationResponse,
  GuardrailConfig,
} from './contracts/types';
/**
 * Purge all indexed documents
 * POST /api/documents/purge
 */
export async function purgeDocuments(): Promise<PurgeResponse> {
  return apiFetch<PurgeResponse>('/api/documents/purge', {
    method: 'POST',
  });
}

/**
 * Get the authenticated tenant's guardrail configuration.
 * GET /api/guardrails
 */
export async function getGuardrails(): Promise<GuardrailConfig> {
  return apiFetch<GuardrailConfig>('/api/guardrails', { method: 'GET' });
}

/**
 * Create a conversation for the authenticated tenant.
 * POST /api/conversations
 */
export async function createConversation(payload: ConversationCreate = {}): Promise<Conversation> {
  return apiFetch<Conversation>('/api/conversations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

/**
 * List conversations for the authenticated tenant.
 * GET /api/conversations?limit=<limit>
 */
export async function listConversations(limit = 50): Promise<ConversationListResponse> {
  return apiFetch<ConversationListResponse>(`/api/conversations?limit=${encodeURIComponent(limit)}`, {
    method: 'GET',
  });
}

/**
 * Get one conversation and its messages.
 * GET /api/conversations/{conversation_id}
 */
export async function getConversation(conversationId: number): Promise<Conversation> {
  return apiFetch<Conversation>(`/api/conversations/${conversationId}`, { method: 'GET' });
}

/**
 * Add a message to a conversation.
 * POST /api/conversations/{conversation_id}/messages
 */
export async function addConversationMessage(
  conversationId: number,
  payload: MessageCreate,
): Promise<Message> {
  return apiFetch<Message>(`/api/conversations/${conversationId}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

/**
 * Delete a conversation.
 * DELETE /api/conversations/{conversation_id}
 */
export async function deleteConversation(conversationId: number): Promise<DeleteConversationResponse> {
  return apiFetch<DeleteConversationResponse>(`/api/conversations/${conversationId}`, {
    method: 'DELETE',
  });
}

// Get API base URL from environment variable
// Defaults to localhost for local dev
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

/**
 * Get stored JWT token from localStorage
 */
function getToken(): string | null {
  const token = localStorage.getItem('ragify_jwt');
  if (!token) return null;

  try {
    const parts = token.split('.');
    if (parts.length === 3) {
      const payload = JSON.parse(atob(parts[1]));
      if (payload.exp && payload.exp * 1000 < Date.now()) {
        localStorage.removeItem('ragify_jwt');
        return null;
      }
    }
  } catch (e) {
    // ignore parse errors, let backend handle invalid tokens
  }
  return token;
}

/**
 * Generic fetch wrapper with error handling
 */
async function apiFetch<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const error: ErrorResponse = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Login with username and password
 * POST /api/login
 */
export async function login(credentials: LoginRequest): Promise<LoginResponse> {
  return apiFetch<LoginResponse>('/api/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(credentials),
  });
}

/**
 * List all documents for current tenant
 * GET /api/documents
 */
export async function listDocuments(): Promise<DocumentsListResponse> {
  return apiFetch<DocumentsListResponse>('/api/documents', {
    method: 'GET',
  });
}

/**
 * Upload documents
 * POST /api/upload
 */
export async function uploadDocuments(files: File[]): Promise<UploadResponse> {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });

  const token = getToken();
  const response = await fetch(`${API_BASE_URL}/api/upload`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: formData,
  });

  if (!response.ok) {
    const error: ErrorResponse = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}
