import { useEffect } from 'react';
import type { DebugInfo } from '../contracts/types';
import './DebugDrawer.css';

interface DebugDrawerProps {
  isOpen: boolean;
  onToggle: () => void;
  debugInfo: DebugInfo | null;
}

export default function DebugDrawer({ isOpen, onToggle, debugInfo }: DebugDrawerProps) {
  // Keyboard shortcut: Ctrl+D
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === 'd') {
        e.preventDefault();
        onToggle();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onToggle]);

  if (!isOpen) {
    // Hidden trigger link (small, unobtrusive)
    return (
      <button 
        className="debug-trigger" 
        onClick={onToggle}
        title="Toggle debug drawer (Ctrl+D)"
        aria-label="Toggle debug drawer"
      >
        ⚙
      </button>
    );
  }

  return (
    <div className="debug-drawer-overlay">
      <div className="debug-drawer">
        <div className="debug-drawer-header">
          <h3>🛠 Debug Information</h3>
          <button 
            className="debug-close" 
            onClick={onToggle}
            aria-label="Close debug drawer"
          >
            ✕
          </button>
        </div>

        {!debugInfo && (
          <div className="debug-empty">
            No debug data yet. Run a query to see debug information.
          </div>
        )}

        {debugInfo && (
          <div className="debug-content">
            <div className="debug-section">
              <h4>Request Metadata</h4>
              <dl className="debug-list">
                {debugInfo.request_id && (
                  <>
                    <dt>Request ID:</dt>
                    <dd><code>{debugInfo.request_id}</code></dd>
                  </>
                )}
                {debugInfo.tenant_id && (
                  <>
                    <dt>Tenant ID:</dt>
                    <dd><code>{debugInfo.tenant_id}</code></dd>
                  </>
                )}
              </dl>
            </div>

            <div className="debug-section">
              <h4>Collection Info</h4>
              <dl className="debug-list">
                {debugInfo.collection_name && (
                  <>
                    <dt>Collection Name:</dt>
                    <dd><code>{debugInfo.collection_name}</code></dd>
                  </>
                )}
                {debugInfo.collection_count !== undefined && debugInfo.collection_count !== null && (
                  <>
                    <dt>Collection Count:</dt>
                    <dd><code>{debugInfo.collection_count}</code></dd>
                  </>
                )}
              </dl>
            </div>

            <div className="debug-section">
              <h4>Retrieval Stats</h4>
              <dl className="debug-list">
                <dt>Evidence Count:</dt>
                <dd><code>{debugInfo.evidence_count}</code></dd>
                
                <dt>Sources Count:</dt>
                <dd><code>{debugInfo.sources_count}</code></dd>

                {debugInfo.retrieved_count !== undefined && debugInfo.retrieved_count !== null && (
                  <>
                    <dt>Retrieved Count:</dt>
                    <dd><code>{debugInfo.retrieved_count}</code></dd>
                  </>
                )}

                {debugInfo.selected_count !== undefined && debugInfo.selected_count !== null && (
                  <>
                    <dt>Selected Count:</dt>
                    <dd><code>{debugInfo.selected_count}</code></dd>
                  </>
                )}
              </dl>
            </div>

            {debugInfo.grounding_gate && (
              <div className="debug-section">
                <h4>Grounding Gate</h4>
                <dl className="debug-list">
                  {debugInfo.grounding_gate.failed_check !== undefined && (
                    <>
                      <dt>Failed Check:</dt>
                      <dd>
                        <code className={debugInfo.grounding_gate.failed_check ? 'debug-error' : 'debug-success'}>
                          {String(debugInfo.grounding_gate.failed_check)}
                        </code>
                      </dd>
                    </>
                  )}
                  {debugInfo.grounding_gate.refusal_reason && (
                    <>
                      <dt>Refusal Reason:</dt>
                      <dd><code className="debug-error">{debugInfo.grounding_gate.refusal_reason}</code></dd>
                    </>
                  )}
                </dl>
              </div>
            )}

            {debugInfo.top10_scores && debugInfo.top10_scores.length > 0 && (
              <div className="debug-section">
                <h4>Top 10 Scores</h4>
                <div className="debug-scores">
                  {debugInfo.top10_scores.map((score, idx) => (
                    <span key={idx} className="debug-score">
                      {idx + 1}: {score.toFixed(4)}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {debugInfo.doc_ids_filter && debugInfo.doc_ids_filter.length > 0 && (
              <div className="debug-section">
                <h4>Doc IDs Filter</h4>
                <code className="debug-code-block">
                  {JSON.stringify(debugInfo.doc_ids_filter, null, 2)}
                </code>
              </div>
            )}
          </div>
        )}

        <div className="debug-drawer-footer">
          <small>Press <kbd>Ctrl+D</kbd> to toggle</small>
        </div>
      </div>
    </div>
  );
}
