import { useState, useRef, useEffect } from 'react';
import type { EvidenceItem, SourceItem } from '../contracts/types';
import './EvidencePanel.css';

interface EvidencePanelProps {
  evidence: EvidenceItem[];
  query: string;
  refused?: boolean;
  sources?: SourceItem[];
}

/**
 * Highlight query tokens in text (case-insensitive)
 */
function highlightText(text: string, query: string): React.ReactNode[] {
  if (!query.trim()) return [text];

  // Extract meaningful tokens from query (ignore common words)
  const stopWords = new Set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'what', 'when', 'where', 'who', 'how', 'why']);
  const tokens = query
    .toLowerCase()
    .split(/\s+/)
    .filter(t => t.length > 2 && !stopWords.has(t));

  if (tokens.length === 0) return [text];

  // Create regex pattern for all tokens
  const pattern = new RegExp(`(${tokens.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'gi');
  const parts = text.split(pattern);

  return parts.map((part, idx) => {
    const isMatch = tokens.some(token => part.toLowerCase() === token);
    return isMatch ? (
      <mark key={idx} className="highlight">{part}</mark>
    ) : (
      <span key={idx}>{part}</span>
    );
  });
}

function detectAnchorType(text: string): 'time' | 'wifi' | 'none' {
  const t = (text || '').toLowerCase();

  // Time: 8:00 AM / 8 AM / 12:30 pm
  const timeRe = /\b\d{1,2}(:\d{2})?\s*(a\.?m\.?|p\.?m\.?)/i;
  if (timeRe.test(t)) return 'time';

  // WiFi: ssid/password patterns
  const wifiRe = /\b(wifi|wi-fi|ssid|password)\b/i;
  if (wifiRe.test(t)) return 'wifi';

  return 'none';
}

function anchorBadgeLabel(type: 'time' | 'wifi' | 'none'): string | null {
  if (type === 'time') return 'Time anchor detected';
  if (type === 'wifi') return 'WiFi anchor detected';
  return null;
}


/**
 * (countMatches removed — was unused)
 */

interface EvidenceItemComponentProps {
  evidence: EvidenceItem;
  index: number;
  query: string;
}

function EvidenceItemComponent({ evidence, index, query }: EvidenceItemComponentProps) {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(evidence.snippet);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  return (
    <div className="evidence-item">
      <div className="evidence-header" onClick={() => setExpanded(!expanded)}>
        <div className="evidence-title">
          {evidence.heading && <span className="evidence-heading">{evidence.heading}</span>}
          {!evidence.heading && <span className="evidence-heading">Evidence {index + 1}</span>}
          {(() => {
            const anchorType = detectAnchorType(evidence.snippet);
            const label = anchorBadgeLabel(anchorType);
            return label ? <span className="match-badge">{label}</span> : null;
          })()}
        </div>
        <div className="evidence-actions">
          <button 
            className="copy-button" 
            onClick={handleCopy}
            aria-label="Copy snippet"
            title="Copy to clipboard"
          >
            {copied ? '✓' : '📋'}
          </button>
          <button className="expand-toggle" aria-label={expanded ? 'Collapse' : 'Expand'}>
            {expanded ? '−' : '+'}
          </button>
        </div>
      </div>

      {expanded && (
        <div className="evidence-body">
          <div className="evidence-snippet">
            {highlightText(evidence.snippet, query)}
          </div>
          <div className="evidence-meta">
            Chunk: {evidence.chunk_id}
            {evidence.doc_id && ` | Doc ID: ${evidence.doc_id}`}
          </div>
        </div>
      )}
    </div>
  );
}

export default function EvidencePanel({ evidence, query, refused, sources }: EvidencePanelProps) {
  // Demo-safe validation: check for malformed evidence items
  const malformed = evidence.some(ev => !ev.snippet || !ev.chunk_id);
  if (malformed && import.meta.env.DEV) {
    // In dev mode still throw to surface issues during development
    throw new Error('Evidence item missing snippet or chunk_id: ' + JSON.stringify(evidence.find(ev => !ev.snippet || !ev.chunk_id)));
  }
  // If refused, show banner and hide evidence
  if (refused) {
    return (
      <div className="evidence-section">
        <div className="evidence-refusal-banner">
          <div className="refusal-icon">🚫</div>
          <div className="refusal-text">
            <strong>No evidence available</strong>
            <p>The query could not be answered from the uploaded documents.</p>
          </div>
        </div>
      </div>
    );
  }

  // If no evidence and not refused, show calm empty state
  if (evidence.length === 0) {
    return (
      <div className="evidence-section">
        <h3>Evidence (0)</h3>
        <div className="evidence-empty">No evidence returned.</div>
      </div>
    );
  }

  // Collapse/expand state for evidence list
  const [showAll, setShowAll] = useState(false);
  const sectionRef = useRef<HTMLDivElement | null>(null);
  const prevScrollTopRef = useRef<number | null>(null);
  const prevScrollElementRef = useRef<HTMLElement | null>(null);

  const getScrollableAncestor = (el: HTMLElement | null): HTMLElement | null => {
    let cur: HTMLElement | null = el;
    while (cur) {
      if (cur.scrollHeight > cur.clientHeight) return cur;
      cur = cur.parentElement as HTMLElement | null;
    }
    return el;
  };

  const handleToggleShowAll = () => {
    const container = getScrollableAncestor(sectionRef.current);
    if (container) {
      prevScrollTopRef.current = container.scrollTop;
      prevScrollElementRef.current = container;
    }
    setShowAll(s => !s);
  };

  // Restore scroll position after toggle to avoid jumping
  useEffect(() => {
    if (prevScrollElementRef.current && prevScrollTopRef.current != null) {
      const elem = prevScrollElementRef.current;
      const top = prevScrollTopRef.current;
      requestAnimationFrame(() => {
        try { elem.scrollTop = top; } catch {};
        prevScrollTopRef.current = null;
        prevScrollElementRef.current = null;
      });
    }
  }, [showAll]);

  // Deduplicate sources by filename
  const uniqueSources = sources ? Array.from(
    new Map(sources.map(src => [src.filename, src])).values()
  ) : [];

  // Determine which indices to show to preserve original ordering
  const visibleIndices = showAll
    ? evidence.map((_, i) => i)
    : evidence.length === 1
      ? [0]
      : [0, 1];

  return (
    <div className="evidence-section" ref={sectionRef}>
      <h3>Evidence ({evidence.length})</h3>
      {malformed && !import.meta.env.DEV && (
        <div className="evidence-malformed">Evidence payload malformed.</div>
      )}
      <div className="evidence-toggle-control">
        {!showAll ? (
          <button onClick={handleToggleShowAll}>Show all ({evidence.length})</button>
        ) : (
          <button onClick={handleToggleShowAll}>Show less</button>
        )}
      </div>

      <div className="evidence-list">
        {visibleIndices.map((origIdx) => (
          <EvidenceItemComponent
            key={origIdx}
            evidence={evidence[origIdx]}
            index={origIdx}
            query={query}
          />
        ))}
      </div>

      {uniqueSources.length > 0 && (
        <div className="sources-list-section">
          <h4>Sources</h4>
          <ul className="sources-list">
            {uniqueSources.map((src, idx) => (
              <li key={idx} className="source-item">
                📄 {src.filename}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
