/**
 * Server-Sent Events (SSE) streaming client for RAGify /api/query endpoint
 * Implements exact SSE parsing as described in src/contracts/sse.md
 * 
 * Event flow:
 * 1. debug event (always first)
 * 2. token events (streamed tokens, or skipped if refused)
 * 3. final event (always last)
 */

import type {
  QueryRequest,
  QueryFinalResponse,
  DebugInfo,
} from './contracts/types';

// Get API base URL from environment variable
// Defaults to localhost for local dev
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Get stored JWT token from localStorage
 */
function getToken(): string | null {
  return localStorage.getItem('ragify_jwt');
}

/**
 * SSE event handlers
 */
export interface SSEEventHandlers {
  onDebug?: (debugInfo: DebugInfo) => void;
  onToken?: (token: string) => void;
  onFinal?: (response: QueryFinalResponse) => void;
  onError?: (error: Error) => void;
}

/**
 * Query documents with SSE streaming
 * 
 * Parses SSE exactly as specified in sse.md:
 * - event: <type>
 * - data: <json>
 * 
 * Event types:
 * - debug: DebugInfo (always first)
 * - token: { t: string } (streamed answer)
 * - final: QueryFinalResponse (always last)
 * 
 * @param request Query request body
 * @param handlers Event handlers for different event types
 * @returns Object with abort function and done promise (resolves on final, rejects on error/timeout/cancel)
 */
export function queryWithSSE(
  request: QueryRequest,
  handlers: SSEEventHandlers
): { abort: () => void; done: Promise<QueryFinalResponse> } {
  const token = getToken();
  if (!token) {
    const error = new Error('No authentication token found');
    handlers.onError?.(error);
    return { abort: () => {}, done: Promise.reject(error) };
  }

  const controller = new AbortController();
  
  // Promise resolvers for done promise
  let resolveDone: (value: QueryFinalResponse) => void;
  let rejectDone: (error: Error) => void;
  const done = new Promise<QueryFinalResponse>((resolve, reject) => {
    resolveDone = resolve;
    rejectDone = reject;
  });
  
  // Extended timeout to match Vite proxy config (120s)
  const timeoutId = setTimeout(() => {
    const error = new Error('Query timeout after 120 seconds. Please try again or check your backend connection.');
    controller.abort();
    handlers.onError?.(error);
    rejectDone(error);
  }, 120000); // 120 second timeout (matches vite.config.ts proxy timeout)

  const executeQuery = async () => {
    try {
      // Handle non-streaming request
      if (request.stream === false) {
        const response = await fetch(`${API_BASE_URL}/api/query`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ ...request, stream: false }),
          signal: controller.signal,
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: response.statusText }));
          throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        const finalData = await response.json() as QueryFinalResponse;
        handlers.onFinal?.(finalData);
        resolveDone(finalData);
        return;
      }

      // Streaming request (original logic)
      const response = await fetch(`${API_BASE_URL}/api/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
          // SSE-specific headers for better proxy compatibility
          'Accept': 'text/event-stream',
          'Cache-Control': 'no-cache',
        },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    if (!response.body) {
      throw new Error('No response body');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    // Parse SSE stream block by block (spec-safe approach)
    // Each event block is separated by blank line (\n\n)
    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      // Normalize line endings and append to buffer
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      
      // Split by double newline (event block separator)
      const blocks = buffer.split('\n\n');
      // Keep the last incomplete block in buffer
      buffer = blocks.pop() || '';

      // Process each complete event block
      for (const block of blocks) {
        if (!block.trim()) continue; // Skip empty blocks

        const lines = block.split('\n');
        let eventType = 'message'; // Default event type per SSE spec
        const dataLines: string[] = [];

        // Parse block lines
        for (const line of lines) {
          if (line.startsWith(':')) {
            // SSE comment/heartbeat - ignore
            continue;
          } else if (line.startsWith('event:')) {
            eventType = line.substring(6).trim();
          } else if (line.startsWith('data:')) {
            // data: prefix, rest is payload (may be empty after colon)
            dataLines.push(line.substring(5).trimStart());
          }
          // Ignore other field types (id:, retry:, etc.)
        }

        // Join multiple data lines with newline per SSE spec
        const dataStr = dataLines.join('\n');
        if (!dataStr) continue; // Skip blocks with no data

        try {
          const data = JSON.parse(dataStr);

          // Dispatch based on event type
          if (eventType === 'debug') {
            handlers.onDebug?.(data as DebugInfo);
          } else if (eventType === 'token') {
            // Token event has shape { t: string }
            if ('t' in data && typeof data.t === 'string') {
              handlers.onToken?.(data.t);
            }
          } else if (eventType === 'final') {
            // Final event - end of stream
            const finalData = data as QueryFinalResponse;
            handlers.onFinal?.(finalData);
            resolveDone(finalData);
            // Close the stream promptly (best effort)
            reader.cancel().catch(() => {});
            return; // Stop processing
          } else if (eventType === 'error') {
            // Error event
            handlers.onError?.(new Error(data.message || 'Stream error'));
          }
        } catch (parseError) {
          console.error(`Failed to parse SSE event '${eventType}':`, parseError);
          const error = new Error('Failed to parse SSE data');
          handlers.onError?.(error);
          rejectDone(error);
        }
      }
    }

      // If we exit the loop without receiving a final event, it's an error
      const error = new Error('Stream ended without final event');
      handlers.onError?.(error);
      rejectDone(error);
    } catch (error) {
      const err = error instanceof Error && error.name === 'AbortError'
        ? new Error('Request cancelled')
        : (error as Error);
      handlers.onError?.(err);
      rejectDone(err);
    } finally {
      clearTimeout(timeoutId);
    }
  };

  executeQuery();

  return {
    abort: () => {
      controller.abort();
    },
    done,
  };
}

/**
 * React hook for SSE query streaming
 * 
 * Example usage:
 * ```tsx
 * const { query, cancel, streaming, answer, debugInfo, finalResponse, error } = useSSEQuery();
 * 
 * const handleQuery = () => {
 *   query({ question: "What is the policy?" });
 * };
 * ```
 */
export function useSSEQuery() {
  const [streaming, setStreaming] = React.useState(false);
  const [answer, setAnswer] = React.useState('');
  const [debugInfo, setDebugInfo] = React.useState<DebugInfo | null>(null);
  const [finalResponse, setFinalResponse] = React.useState<QueryFinalResponse | null>(null);
  const [error, setError] = React.useState<Error | null>(null);
  const abortRef = React.useRef<(() => void) | null>(null);

  const query = React.useCallback((request: QueryRequest) => {
    setStreaming(true);
    setAnswer('');
    setDebugInfo(null);
    setFinalResponse(null);
    setError(null);

    const { abort } = queryWithSSE(request, {
      onDebug: (debug) => {
        setDebugInfo(debug);
      },
      onToken: (token) => {
        setAnswer((prev) => prev + token);
      },
      onFinal: (final) => {
        setFinalResponse(final);
        setStreaming(false);
        abortRef.current = null;
      },
      onError: (err) => {
        setError(err);
        setStreaming(false);
        abortRef.current = null;
      },
    });

    abortRef.current = abort;
  }, []);

  const cancel = React.useCallback(() => {
    if (abortRef.current) {
      abortRef.current();
      abortRef.current = null;
    }
    setStreaming(false);
  }, []);

  return {
    query,
    cancel,
    streaming,
    answer,
    debugInfo,
    finalResponse,
    error,
  };
}

// Import React for the hook
import * as React from 'react';
