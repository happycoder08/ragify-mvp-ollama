import { useState, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { queryWithSSE } from '../sse';
import { listDocuments } from '../api';
import type { DebugInfo, DocumentRecord } from '../contracts/types';
import { getSelectedDocIds } from '../utils/documentSelection';
import { computeAnswerMode, labelForMode, tooltipForModeWithContext } from '../utils/computeAnswerMode';
import EvidencePanel from './EvidencePanel';
import DebugDrawer from './DebugDrawer';
import './Query.css';

type ConversationTurn = {
  user: string;
  assistant?: string;
};

const CONVERSATION_ID_KEY = 'ragify_conversation_id';
const MAX_MESSAGE_CHARS = 800;

function clampMessageContent(text: string): string {
  if (!text) return '';
  return text.length > MAX_MESSAGE_CHARS ? text.slice(0, MAX_MESSAGE_CHARS) : text;
}

function getOrCreateConversationId(): number {
  if (typeof window === 'undefined' || !window.localStorage) {
    return Date.now();
  }

  const existing = window.localStorage.getItem(CONVERSATION_ID_KEY);
  if (existing) {
    const parsed = Number(existing);
    if (!Number.isNaN(parsed) && parsed > 0) {
      return parsed;
    }
  }

  const newId = Date.now() + Math.floor(Math.random() * 1000);
  window.localStorage.setItem(CONVERSATION_ID_KEY, String(newId));
  return newId;
}

type BuildCopyTextArgs = {
  lastQuery: string;
  answerModeLabel: string;
  answerText: string;
  sources: any[];
  evidence: any[];
};

function buildCopyText({ lastQuery, answerModeLabel, answerText, sources, evidence }: BuildCopyTextArgs): string {
  const lines: string[] = [];

  lines.push(`Question: ${lastQuery || ''}`);
  lines.push(`Answer Mode: ${answerModeLabel || ''}`);
  lines.push(`Answer: ${answerText || ''}`);
  lines.push('');

  lines.push('Sources:');
  const filenames = (sources || [])
    .map((s: any) => s && s.filename)
    .filter((name: any) => Boolean(name));
  if (filenames.length > 0) {
    filenames.forEach((name: string) => lines.push(name));
  } else {
    lines.push('None');
  }
  lines.push('');

  lines.push('Evidence:');
  const evidenceItems = Array.isArray(evidence) ? evidence : [];
  if (evidenceItems.length === 0) {
    lines.push('None');
  } else {
    evidenceItems.forEach((ev: any, index: number) => {
      if (index > 0) {
        lines.push('---');
      }
      const heading = ev && ev.heading ? String(ev.heading) : `Evidence ${index + 1}`;
      const chunkId = ev && ev.chunk_id ? String(ev.chunk_id) : 'unknown';
      const snippet = ev && ev.snippet ? String(ev.snippet) : '';

      lines.push(`[${index + 1}] ${heading} (chunk_id=${chunkId})`);
      if (snippet) {
        lines.push(snippet);
      }
    });
  }

  return lines.join('\n');
}

async function copyTextWithFallback(text: string): Promise<void> {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }
    throw new Error('Clipboard API not available');
  } catch {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.top = '-9999px';
    textarea.style.left = '-9999px';
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();

    try {
      const successful = document.execCommand('copy');
      document.body.removeChild(textarea);
      if (!successful) {
        throw new Error('Fallback copy failed');
      }
    } catch (err) {
      document.body.removeChild(textarea);
      throw err;
    }
  }
}

export function buildDisambiguatedQuestion(originalQuestion: string, option: string): string {
  return `${originalQuestion} (${option})`;
}

function formatClarificationType(type?: string): string | null {
  if (!type) return null;
  return type.replace(/_/g, ' ').trim();
}

type ClarificationData = {
  type?: string;
  question?: string;
  options?: string[];
};

type ClarificationCardProps = {
  clarification: ClarificationData;
  onSelect: (option: string) => void;
  disabled: boolean;
};

function ClarificationCard({ clarification, onSelect, disabled }: ClarificationCardProps) {
  const typeLabel = formatClarificationType(clarification.type);
  return (
    <div className="clarification-card">
      <div className="clarification-question">
        {clarification.question}
        {typeLabel && <span className="clarification-type">({typeLabel})</span>}
      </div>
      {Array.isArray(clarification.options) && clarification.options.length > 0 && (
        <div className="clarification-options">
          {clarification.options.map((opt) => (
            <button
              key={opt}
              className="clarification-option-btn"
              onClick={() => onSelect(opt)}
              disabled={disabled}
              type="button"
            >
              {opt}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// Demo mode configuration from environment
const isDemoMode = import.meta.env.VITE_DEMO_MODE === 'true';
const DEMO_MODE = isDemoMode;

// Dev tools flag from environment or URL param (hardened for production)
const envShowDevTools = import.meta.env.VITE_SHOW_DEVTOOLS === 'true';
const canUseUrlDebug = import.meta.env.DEV || envShowDevTools;
const urlDebug = canUseUrlDebug && new URLSearchParams(window.location.search).get('debug') === '1';

const SHOW_DEVTOOLS =
  import.meta.env.DEV ||
  envShowDevTools ||
  urlDebug ||
  (globalThis as any).__TEST_SHOW_DEVTOOLS__ === true;

// Simple in-memory cache for document status (10 second TTL)
let docStatusCache: { hasPending: boolean; timestamp: number } | null = null;
const CACHE_TTL = 10000; // 10 seconds

export default function Query() {
  const [question, setQuestion] = useState('');
  const [streaming, setStreaming] = useState(false);
  // State machine states
  const [answerText, setAnswerText] = useState('');
  const answerBufferRef = useRef('');
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null);
  const [evidence, setEvidence] = useState<any[]>([]);
  const [sources, setSources] = useState<any[]>([]);
  const [refused, setRefused] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Dev-only diagnostic banners
  const [devInvariantMsg, setDevInvariantMsg] = useState<string | null>(null);
  const [devMismatchMsg, setDevMismatchMsg] = useState<string | null>(null);
  const [devSuccessMsg, setDevSuccessMsg] = useState<string | null>(null);
  const [hasPendingDocs, setHasPendingDocs] = useState(false);
  const [debugDrawerOpen, setDebugDrawerOpen] = useState(false);
  const [lastQuery, setLastQuery] = useState<string>('');
  const [selectedDocIds, setSelectedDocIds] = useState<number[]>(() => getSelectedDocIds());
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [rawResponse, setRawResponse] = useState<string>('');
  const [useSelectedDocs, setUseSelectedDocs] = useState(false);
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [conversationHistory, setConversationHistory] = useState<ConversationTurn[]>([]);
  const [clarification, setClarification] = useState<ClarificationData | null>(null);
  const [needsClarification, setNeedsClarification] = useState(false);
  
  const [copySuccess, setCopySuccess] = useState(false);
  const [hasFinal, setHasFinal] = useState(false);
  const [responsePipelineMarker, setResponsePipelineMarker] = useState<string | null>(null);
  const abortRef = useRef<(() => void) | null>(null);
  const lastDebugLevelRef = useRef(0);

  // Readiness check: fetch documents on mount and check for pending status
  useEffect(() => {
    const checkReadiness = async () => {
      // Check cache first
      if (docStatusCache && Date.now() - docStatusCache.timestamp < CACHE_TTL) {
        setHasPendingDocs(docStatusCache.hasPending);
        return;
      }

      try {
        const response = await listDocuments();
        const pending = response.documents.some(doc => doc.status === 'pending');

        // Update cache
        docStatusCache = {
          hasPending: pending,
          timestamp: Date.now(),
        };

        setHasPendingDocs(pending);
        setDocuments(response.documents);
      } catch (err) {
        console.error('Failed to check document readiness:', err);
      }
    };

    checkReadiness();

    // Refresh selected doc IDs when page loads (in case changed on Docs page)
    setSelectedDocIds(getSelectedDocIds());

    // Initialize or restore conversation id on first mount
    const id = getOrCreateConversationId();
    setConversationId(id);
  }, []);

  const executeQuery = (questionText: string) => {
    if (!questionText.trim()) return;

    const ensuredConversationId = conversationId ?? getOrCreateConversationId();
    if (conversationId == null) {
      setConversationId(ensuredConversationId);
    }

    const trimmedQuestion = questionText.trim();
    const cappedQuestion = clampMessageContent(trimmedQuestion);

    // Capture snapshot of existing history for this request (exclude new question)
    const historySnapshot = conversationHistory;

    setStreaming(true);
    setHasFinal(false);
    setAnswerText('');
    answerBufferRef.current = '';
    setDebugInfo(null);
    setEvidence([]);
    setSources([]);
    setRefused(false);
    setError(null);
    setRawResponse('');
    setLastQuery(trimmedQuestion);
    setClarification(null);
    setNeedsClarification(false);

    // Update in-memory conversation history with new user turn (for future requests)
    setConversationHistory(prev => {
      const next = [...prev, { user: clampMessageContent(trimmedQuestion) }];
      const overflow = Math.max(0, next.length - 8);
      return overflow > 0 ? next.slice(overflow) : next;
    });

    const mappedHistory = historySnapshot.flatMap(turn => {
      const messages: { role: 'user' | 'assistant'; content: string }[] = [];
      if (turn.user) {
        messages.push({ role: 'user', content: clampMessageContent(turn.user) });
      }
      if (turn.assistant) {
        messages.push({ role: 'assistant', content: clampMessageContent(turn.assistant) });
      }
      return messages;
    });

    const debugLevel = debugDrawerOpen ? 2 : (SHOW_DEVTOOLS ? 1 : 0);
    lastDebugLevelRef.current = debugLevel;
    const queryRequest = {
      question: cappedQuestion,
      mode: 'full' as const,
      top_k: 4,
      debug: debugLevel,
      stream: false,
      conversation_id: ensuredConversationId,
      ...(mappedHistory.length > 0 ? { conversation_history: mappedHistory } : {}),
      ...(useSelectedDocs && selectedDocIds.length > 0 ? { doc_ids: selectedDocIds } : {}),
    };

    // DEV logging: print outgoing query payload and doc_ids status
    if (import.meta.env.DEV) {
      if ('doc_ids' in queryRequest) {
        console.log('[Query] Sending request with doc_ids:', (queryRequest as any).doc_ids, queryRequest);
      } else {
        console.log('[Query] Sending request for ALL docs (no doc_ids):', queryRequest);
      }
    }

    const { abort } = queryWithSSE(
      queryRequest,
      {
        // Rule 1: On 'debug' event: setDebugInfo(data) only. Do NOT set answer.
        onDebug: (data) => {
          console.log('[SSE debug] Setting debugInfo:', data);
          setDebugInfo(data);
        },
        // Rule 2: On 'token' event: append data.t to answerBufferRef and setAnswerText(answerBufferRef) for live streaming.
        onToken: (token) => {
          answerBufferRef.current += token;
          setAnswerText(answerBufferRef.current);
        },
        // Rule 3: On 'final' event: set all final state
        onFinal: (data) => {
          console.log('[SSE final] Setting final state with answer:', data.answer);
          setRawResponse(JSON.stringify(data, null, 2));
          setRefused(data.refused);
          setAnswerText(data.answer);
          setEvidence(data.evidence);
          setSources(data.sources);
          setDebugInfo(prev => data.debug_info ?? prev);

          const debugPm = (data as any).debug_info?.pipeline_marker;
          const debugNeedsClarification = (data as any).debug_info?.needs_clarification === true;
          const isClarification =
            (data as any).pipeline_marker === 'CLARIFICATION_REQUIRED' ||
            (data as any).needs_clarification === true ||
            debugPm === 'CLARIFICATION_REQUIRED' ||
            debugNeedsClarification;
          setNeedsClarification(isClarification);
          if (isClarification && (data as any).clarification) {
            setClarification((data as any).clarification);
          } else {
            setClarification(null);
          }
          const clarificationQuestion = (data as any).clarification?.question;
          const assistantContent =
            isClarification && clarificationQuestion ? clarificationQuestion : data.answer ?? '';

          // Update conversation history with assistant response for most recent user turn
          setConversationHistory(prev => {
            if (prev.length === 0) {
              const singleTurn: ConversationTurn = {
                user: clampMessageContent(trimmedQuestion),
                assistant: clampMessageContent(assistantContent),
              };
              return [singleTurn];
            }

            const updated = [...prev];
            const lastTurn = updated[updated.length - 1];
            if (lastTurn && !lastTurn.assistant) {
              updated[updated.length - 1] = {
                ...lastTurn,
                assistant: clampMessageContent(assistantContent),
              };
            } else {
              updated.push({
                user: clampMessageContent(trimmedQuestion),
                assistant: clampMessageContent(assistantContent),
              });
            }

            const overflow = Math.max(0, updated.length - 8);
            return overflow > 0 ? updated.slice(overflow) : updated;
          });

          // capture pipeline_marker if present in response (defensive)
          try {
            setResponsePipelineMarker(((data as any).pipeline_marker as string) ?? null);
          } catch {
            setResponsePipelineMarker(null);
          }
          setStreaming(false);
          abortRef.current = null;

          // DEV validation: check invariants (allow empty evidence/sources for CLARIFICATION_REQUIRED)
          const pm = ((data as any).pipeline_marker as string) ?? null;
          const allowEmptyEvidenceSources = isClarification;
          let nextDevInvariantMsg: string | null = null;
          let nextDevSuccessMsg: string | null = null;

          if (!allowEmptyEvidenceSources) {
            if (!data.evidence || !data.sources || data.evidence.length === 0 || data.sources.length === 0) {
              nextDevInvariantMsg = 'DEV ERROR: final payload missing evidence or sources';
            }
          }

          if (!nextDevInvariantMsg && SHOW_DEVTOOLS && !DEMO_MODE && pm === 'LLM_VALIDATED') {
            const evidenceText = (data.evidence || [])
              .map(ev => (ev && ev.snippet ? String(ev.snippet) : ''))
              .join(' ');
            const evidenceHasInteger = /\b\d+\b/.test(evidenceText);
            const answerHasInteger = /\b\d+\b/.test(String(data.answer ?? ''));
            if (evidenceHasInteger && !answerHasInteger) {
              nextDevInvariantMsg = 'Validated answer missing numeric evidence';
            }
          }

          if (SHOW_DEVTOOLS && !DEMO_MODE && pm === 'EXTRACTOR_FALLBACK' && lastDebugLevelRef.current > 0) {
            nextDevSuccessMsg = '🛡️ Hallucination prevented: Fallback to strict extraction.';
          }

          // DEV validation: check for answer/evidence mismatch
          if (data.answer && data.evidence && data.evidence.length > 0) {
            const answerLower = data.answer.toLowerCase();
            const hasTimeInAnswer = /\b\d{1,2}:\d{2}\b/.test(answerLower) || /\b\d{1,2}(am|pm)\b/i.test(answerLower);
            const hasTimeInEvidence = data.evidence.some(ev => 
              ev.snippet && (/\b\d{1,2}:\d{2}\b/.test(ev.snippet) || /\b\d{1,2}(am|pm)\b/i.test(ev.snippet))
            );
            if (hasTimeInAnswer && !hasTimeInEvidence) {
              setDevMismatchMsg('Answer/Evidence mismatch');
            } else {
              setDevMismatchMsg(null);
            }
          } else {
            setDevMismatchMsg(null);
          }

          // DEV validation: check canonical refusal
          const canonicalRefusal = 'The document does not specify this.';
          if (!data.refused && data.answer === canonicalRefusal) {
            nextDevInvariantMsg = 'DEV ERROR: final.answer equals canonical refusal while refused=false';
          }

          setDevInvariantMsg(nextDevInvariantMsg);
          setDevSuccessMsg(nextDevSuccessMsg);
          setHasFinal(true);
        },
        // Rule 4: On 'error': show error state and stop.
        onError: (err) => {
          setError(err.message);
          setStreaming(false);
          abortRef.current = null;
        },
      }
    );

    abortRef.current = abort;
  };

  const handleQuery = (e: React.FormEvent) => {
    e.preventDefault();
    executeQuery(question);
  };

  const handleCancel = () => {
    if (abortRef.current) {
      abortRef.current();
      abortRef.current = null;
    }
    setStreaming(false);
    setAnswerText('');
    answerBufferRef.current = '';
    setDevInvariantMsg(null);
    setDevMismatchMsg(null);
    setDevSuccessMsg(null);
  };

  const handleClear = () => {
    setQuestion('');
    setAnswerText('');
    answerBufferRef.current = '';
    setDebugInfo(null);
    setEvidence([]);
    setSources([]);
    setRefused(false);
    setError(null);
    setRawResponse('');
    setClarification(null);
    setNeedsClarification(false);
    if (abortRef.current) {
      abortRef.current();
      abortRef.current = null;
    }
    setStreaming(false);
    setDevInvariantMsg(null);
    setDevMismatchMsg(null);
    setDevSuccessMsg(null);
    setHasFinal(false);
  };

  const handleRetry = () => {
    if (lastQuery) {
      executeQuery(lastQuery);
    }
  };

  const runDemoQuery = (demoQuestion: string) => {
    setQuestion(demoQuestion);
    setTimeout(() => executeQuery(demoQuestion), 100);
  };

  const handleClarificationOption = (option: string) => {
    if (!lastQuery) return;
    const typeLabel = formatClarificationType(clarification?.type);
    const newQuestion = typeLabel
      ? `${lastQuery} (${typeLabel}: ${option})`
      : buildDisambiguatedQuestion(lastQuery, option);
    setQuestion(newQuestion);
    setTimeout(() => executeQuery(newQuestion), 100);
  };

  // Get scope display text
  const getScopeText = () => {
    if (!useSelectedDocs) return 'All documents';
    if (selectedDocIds.length === 0) return '0 selected';
    const selectedDocs = documents.filter(d => selectedDocIds.includes(d.id));
    if (selectedDocs.length === 1) return selectedDocs[0].filename;
    return `${selectedDocs.length} selected`;
  };

  // Derived answer mode (null-safe)
  const answerMode = computeAnswerMode({
    refused,
    needs_clarification: needsClarification,
    pipeline_marker: responsePipelineMarker ?? undefined,
    debug_info: debugInfo,
  });
  
  const answerModeLabel: string = labelForMode(answerMode);

  const answerModeTooltip = tooltipForModeWithContext(answerMode, {
    pipeline_marker: responsePipelineMarker,
    needs_clarification: needsClarification,
    debug_info: debugInfo,
  });

  return (
    <div className="query-page">
      <div className="query-container">
        <div className="page-guidance">
          <div className="step-header">
            <span className="step-badge">Step 2 of 2</span>
            <h1>Ask Questions</h1>
          </div>
          <p className="step-description">
            Ask questions about your uploaded documents. The AI will search your knowledge base and provide answers with sources.
          </p>
        </div>

        {/* Readiness Banner - Show if documents are still indexing */}
        {hasPendingDocs && (
          <div className="readiness-banner">
            <span className="readiness-icon">⏳</span>
            Indexing in progress — answers may refuse until ready.{' '}
            <Link to="/docs" className="readiness-link">View documents →</Link>
          </div>
        )}

        {/* Demo Mode: Try demo questions */}
        {DEMO_MODE && (
          <div className="demo-questions-card">
            <h3>✨ Try demo questions</h3>
            <div className="demo-buttons">
              <button
                onClick={() => runDemoQuery("What are the main security protocols?")}
                disabled={streaming}
                className="demo-question-btn"
              >
                What are the main security protocols?
              </button>
              <button
                onClick={() => runDemoQuery("Explain the data retention policy")}
                disabled={streaming}
                className="demo-question-btn"
              >
                Explain the data retention policy
              </button>
            </div>
          </div>
        )}

        {/* Top Section: Question Input */}
        {/* Demo Control Bar: scope toggle and selected-docs multi-select */}
        <div style={{ marginBottom: 12 }}>
          <label style={{ marginRight: 12 }}>
            <input
              type="radio"
              name="scope"
              checked={!useSelectedDocs}
              onChange={() => setUseSelectedDocs(false)}
            />{' '}
            All docs
          </label>
          <label>
            <input
              type="radio"
              name="scope"
              checked={useSelectedDocs}
              onChange={() => setUseSelectedDocs(true)}
            />{' '}
            Selected docs
          </label>

          {useSelectedDocs && (
            <div style={{ marginTop: 8 }}>
              <select
                multiple
                size={Math.min(6, Math.max(3, documents.length))}
                value={selectedDocIds.map(String)}
                onChange={(e) => {
                  const opts = Array.from(e.target.selectedOptions).map(o => Number(o.value));
                  setSelectedDocIds(opts);
                }}
                style={{ width: '100%', minWidth: 200 }}
              >
                {documents.map(d => (
                  <option key={d.id} value={d.id}>{d.filename}</option>
                ))}
              </select>
            </div>
          )}
        </div>
        <div className="query-input-card">
          <form onSubmit={handleQuery} className="query-form">
            <div className="form-group">
              <label htmlFor="question">Your Question</label>
              <textarea
                id="question"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="What is the company vacation policy?"
                rows={3}
                disabled={streaming}
                required
              />
            </div>
            <div className="button-group">
              <button type="submit" disabled={streaming || !question.trim()} className="ask-button">
                {streaming ? 'Asking...' : 'Ask'}
              </button>
              {streaming && (
                <button type="button" onClick={handleCancel} className="cancel-button">
                  Cancel
                </button>
              )}
              {(question || answerText || error) && !streaming && (
                <button type="button" onClick={handleClear} className="clear-button">
                  Clear
                </button>
              )}
            </div>
          </form>
        </div>

        {/* Document Scope Indicator */}
        <div className="scope-indicator">
          <span className="scope-label">Scope:</span>
          <span className="scope-value">{getScopeText()}</span>
          <Link to="/docs" className="scope-link">Change →</Link>
        </div>



        {error && (
          <div className="error-message">
            {error}
            {error.toLowerCase().includes('timeout') && lastQuery && (
              <button 
                type="button" 
                onClick={handleRetry} 
                className="retry-button"
                style={{ marginLeft: '12px' }}
              >
                Retry
              </button>
            )}
          </div>
        )}

        {/* Middle & Right: Answer and Evidence */}
        {(streaming || answerText || (hasFinal && needsClarification)) && (
          <div className="results-grid">
            {/* Middle: Answer Panel */}
            <div className="answer-panel">
              <h2>
                Answer
                {hasFinal && (
                  <span
                    title={answerModeTooltip}
                    style={{
                      marginLeft: 8,
                      fontSize: 12,
                      padding: '2px 8px',
                      borderRadius: 9999,
                      background: '#eee',
                      fontWeight: 600,
                      verticalAlign: 'middle',
                    }}
                  >
                    {answerModeLabel}
                  </span>
                )}
              </h2>
              
              {/* Refusal Banner */}
              {refused && (
                <div className="refusal-banner">
                  <div className="refusal-title">⚠️ Answer not found in uploaded documents</div>
                  {/* Note: refusal_reason not available in new state machine */}
                </div>
              )}

              {/* DEV diagnostic banners */}
              {SHOW_DEVTOOLS && !DEMO_MODE && devInvariantMsg && (
                <div className="dev-error-banner" style={{ background: '#ffdddd', padding: 8, marginBottom: 8, borderRadius: 6 }}>
                  <strong>{devInvariantMsg}</strong>
                </div>
              )}
              {SHOW_DEVTOOLS && !DEMO_MODE && devMismatchMsg && (
                <div className="dev-mismatch-banner" style={{ background: '#fff3bf', padding: 8, marginBottom: 8, borderRadius: 6 }}>
                  <strong>{devMismatchMsg}</strong>
                </div>
              )}
              {SHOW_DEVTOOLS && !DEMO_MODE && devSuccessMsg && (
                <div className="dev-success-banner" style={{ background: '#e6ffed', padding: 8, marginBottom: 8, borderRadius: 6 }}>
                  <strong>{devSuccessMsg}</strong>
                </div>
              )}

              <div className="answer-content">
                {answerText}
                {streaming && <span className="cursor">▊</span>}
              </div>

              {/* Clarification Options */}
              {hasFinal && needsClarification && clarification?.question && (
                <ClarificationCard
                  clarification={clarification}
                  onSelect={handleClarificationOption}
                  disabled={streaming}
                />
              )}

              <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
                <button
                  type="button"
                  className="clear-button copy-answer-button"
                  onClick={async () => {
                    const text = buildCopyText({
                      lastQuery,
                      answerModeLabel,
                      answerText,
                      sources: sources || [],
                      evidence: evidence || [],
                    });

                    try {
                      await copyTextWithFallback(text);
                      setCopySuccess(true);
                      setTimeout(() => setCopySuccess(false), 1500);
                    } catch (err) {
                      console.error('Copy failed', err);
                    }
                  }}
                >
                  Copy answer + citations
                </button>
                {copySuccess && (
                  <span style={{ color: 'green', fontSize: 12 }}>Copied!</span>
                )}
              </div>

              {SHOW_DEVTOOLS && !DEMO_MODE && debugInfo && (
                <div className="debug-info">
                  <strong>Debug:</strong> {debugInfo.evidence_count} evidence chunks, {debugInfo.sources_count} sources
                </div>
              )}
            </div>

            {/* Right/Below: Evidence Panel (render only after final) */}
            {hasFinal && !refused && evidence && (
              <div className="evidence-panel-container">
                {evidence.length > 0 && evidence[0].anchor_type && (
                  <div className="anchor-type-banner">
                    {evidence[0].anchor_type === 'WIFI' && '🔗 WIFI ANCHOR DETECTED'}
                    {evidence[0].anchor_type === 'TIME' && '⏰ TIME ANCHOR DETECTED'}
                  </div>
                )}

                <EvidencePanel 
                  evidence={evidence}
                  query={question}
                  refused={refused}
                  sources={sources}
                />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Temporary Debug Panel */}
      {SHOW_DEVTOOLS && !DEMO_MODE && rawResponse && (
        <div style={{ marginTop: '20px', padding: '10px', border: '1px solid #ccc', backgroundColor: '#f9f9f9' }}>
          <h3>Raw Response JSON</h3>
          <pre style={{ whiteSpace: 'pre-wrap', fontSize: '12px' }}>{rawResponse}</pre>
        </div>
      )}

      {/* Debug Drawer */}
      {SHOW_DEVTOOLS && (
        <DebugDrawer
          isOpen={debugDrawerOpen}
          onToggle={() => setDebugDrawerOpen(!debugDrawerOpen)}
          debugInfo={debugInfo}
        />
      )}
    </div>
  );
}
