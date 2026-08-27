/**
 * API client utilities for RAGify backend.
 */

import { getToken } from './auth';
import type {
  LoginRequest,
  LoginResponse,
  QueryRequest,
  DocumentsListResponse,
  UploadResponse,
  SSETokenEvent,
  SSEDebugEvent,
  SSEFinalEvent,
  SSEErrorEvent,
} from '../types/api';

// Get API base URL from environment variable
// Defaults to localhost for local dev
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Generic fetch wrapper with auth header.
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

  const response = await fetch(`${API_URL}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Login with username and password.
 */
export async function login(credentials: LoginRequest): Promise<LoginResponse> {
  return apiFetch<LoginResponse>('/api/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(credentials),
  });
}

/**
 * List documents for current tenant.
 */
export async function listDocuments(): Promise<DocumentsListResponse> {
  return apiFetch<DocumentsListResponse>('/api/documents', {
    method: 'GET',
  });
}

/**
 * Upload documents.
 */
export async function uploadDocuments(files: File[]): Promise<UploadResponse> {
  const formData = new FormData();
  files.forEach((file) => {
    formData.append('files', file);
  });

  const token = getToken();
  const response = await fetch(`${API_URL}/api/upload`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * SSE Event types from backend.
 */
export type SSEEvent =
  | { type: 'token'; data: SSETokenEvent }
  | { type: 'debug'; data: SSEDebugEvent }
  | { type: 'final'; data: SSEFinalEvent }
  | { type: 'error'; data: SSEErrorEvent };

/**
 * Query documents with SSE streaming.
 * 
 * CRITICAL: Event-based SSE parsing with clear stop conditions.
 * - Stops on 'final' event (complete response)
 * - Stops on 'error' event
 * - Timeout after 30 seconds
 * 
 * DO NOT use regex parsing - parse event: and data: lines explicitly.
 */
type QueryWithSSECallbacks = {
  onToken?: (e: SSETokenEvent) => void;
  onDebug?: (e: SSEDebugEvent) => void;
  onFinal?: (e: SSEFinalEvent) => void;
  onSSEErrorEvent?: (e: SSEErrorEvent) => void; // error event coming from server
  onError?: (err: Error) => void;              // transport/parse/etc
  onOpen?: () => void;
  onClose?: () => void;
};

type QueryWithSSEOptions = {
  signal?: AbortSignal;       // caller-controlled abort
  timeoutMs?: number;         // default e.g. 30000
  stopOnFinal?: boolean;      // default true
};

function combineSignals(a?: AbortSignal, b?: AbortSignal): AbortSignal | undefined {
  if (!a) return b;
  if (!b) return a;
  const controller = new AbortController();
  const onAbort = () => controller.abort();
  if (a.aborted || b.aborted) controller.abort();
  a.addEventListener("abort", onAbort, { once: true });
  b.addEventListener("abort", onAbort, { once: true });
  return controller.signal;
}

/**
 * Spec-correct SSE parsing:
 * - Events separated by blank line
 * - Multiple data: lines allowed (join with \n)
 * - Ignores comments
 * - Supports CRLF
 */
export function queryWithSSE(
  request: QueryRequest,
  cb: QueryWithSSECallbacks,
  opts: QueryWithSSEOptions = {}
): { abort: () => void; promise: Promise<void> } {
  const token = getToken();
  const controller = new AbortController();
  const signal = combineSignals(controller.signal, opts.signal);

  const timeoutMs = opts.timeoutMs ?? 30000;
  let timeoutId: number | undefined;
  let stopped = false;

  const abort = () => controller.abort();

  const promise = (async () => {
    if (!token) {
      cb.onError?.(new Error("Missing auth token"));
      return;
    }

    if (timeoutMs > 0) {
      timeoutId = window.setTimeout(() => {
        controller.abort();
        if (!stopped) cb.onError?.(new Error(`Query timeout after ${timeoutMs}ms`));
      }, timeoutMs);
    }

    let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;

    try {
      const response = await fetch(`${API_URL}/api/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "text/event-stream",
          "Cache-Control": "no-cache",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(request),
        signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      if (!response.body) {
        throw new Error("No response body");
      }

      cb.onOpen?.();

      reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");

      let buffer = "";

      // current event frame being built
      let eventName = "";
      let dataLines: string[] = [];

      const dispatch = () => {
        if (stopped) return;
        if (!eventName && dataLines.length === 0) return;

        const dataStr = dataLines.join("\n");
        dataLines = [];

        // Default event type if omitted
        const type = eventName || "message";
        eventName = "";

        if (!dataStr) return;

        let payload: any;
        try {
          payload = JSON.parse(dataStr);
        } catch (e) {
          // Don’t kill the stream on one bad frame — surface it
          if (!stopped) cb.onError?.(new Error(`Failed to parse SSE JSON for event "${type}": ${dataStr.slice(0, 200)}`));
          return;
        }

        if (type === "token") cb.onToken?.(payload as SSETokenEvent);
        else if (type === "debug") cb.onDebug?.(payload as SSEDebugEvent);
        else if (type === "final") cb.onFinal?.(payload as SSEFinalEvent);
        else if (type === "error") cb.onSSEErrorEvent?.(payload as SSEErrorEvent);
        // else ignore unknown events

        if ((opts.stopOnFinal ?? true) && type === "final") {
          // hard stop: cancel reader + abort fetch
          stopped = true;
          try { reader?.cancel(); } catch {}
          controller.abort();
        }

        if (type === "error") {
          // server signaled an error; stop and surface
          try { reader?.cancel(); } catch {}
          controller.abort();
          const detail = (payload && (payload.detail || payload.message)) ? String(payload.detail || payload.message) : "Query error";
          if (!stopped) cb.onError?.(new Error(detail));
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Split into lines (support \n, keep \r out)
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (let rawLine of lines) {
          const line = rawLine.replace(/\r$/, "");

          // blank line = end of event frame
          if (line === "") {
            dispatch();
            continue;
          }

          // comment/keepalive
          if (line.startsWith(":")) continue;

          if (line.startsWith("event:")) {
            eventName = line.slice(6).trim();
            continue;
          }

          if (line.startsWith("data:")) {
            // spec: "data:" may be followed by optional space
            const v = line.slice(5);
            dataLines.push(v.startsWith(" ") ? v.slice(1) : v);
            continue;
          }

          // Optional: handle id:, retry: if you ever need it
        }
      }

      // flush any last frame if stream ended without blank line
      dispatch();
      stopped = true;

    } catch (err: any) {
      if (err?.name === "AbortError") {
        // treat as normal cancellation unless timeout already reported
        // (avoid double-reporting)
        // you can choose to call onClose only
      } else {
        if (!stopped) cb.onError?.(err instanceof Error ? err : new Error(String(err)));
      }
    } finally {
      if (timeoutId) window.clearTimeout(timeoutId);
      cb.onClose?.();
      try { reader?.cancel(); } catch {}
    }
  })();

  return { abort, promise };
}