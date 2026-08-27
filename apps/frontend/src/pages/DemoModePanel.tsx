import { useState, useEffect } from 'react';
import { listDocuments } from '../api';
import './DemoModePanel.css';

interface DemoModePanelProps {
  onRunDemo: (question: string) => void;
  onClear: () => void;
  disabled: boolean;
}

const DEMO_Q1 = import.meta.env.VITE_DEMO_Q1 || '';
const DEMO_Q2 = import.meta.env.VITE_DEMO_Q2 || '';

export default function DemoModePanel({ onRunDemo, onClear, disabled }: DemoModePanelProps) {
  const [hasIndexedDocs, setHasIndexedDocs] = useState(false);
  const [checking, setChecking] = useState(true);
  const [configWarning, setConfigWarning] = useState<string | null>(null);

  useEffect(() => {
    // Check for missing environment variables
    const warnings: string[] = [];
    if (!DEMO_Q1) warnings.push('VITE_DEMO_Q1');
    if (!DEMO_Q2) warnings.push('VITE_DEMO_Q2');
    
    if (warnings.length > 0) {
      setConfigWarning(`Missing env vars: ${warnings.join(', ')}`);
    }

    // Check document readiness
    const checkDocuments = async () => {
      try {
        const response = await listDocuments();
        const indexed = response.documents.some(doc => doc.status === 'indexed');
        setHasIndexedDocs(indexed);
      } catch (err) {
        console.error('Failed to check documents for demo readiness:', err);
      } finally {
        setChecking(false);
      }
    };

    checkDocuments();
  }, []);

  const isDemoReady = hasIndexedDocs && !checking;

  return (
    <div className="demo-mode-panel">
      <h3>Demo Mode</h3>
      
      {/* Readiness Indicator */}
      <div className={`demo-readiness ${isDemoReady ? 'ready' : 'not-ready'}`}>
        {checking ? (
          <span>⏳ Checking readiness...</span>
        ) : isDemoReady ? (
          <span>✓ Demo ready</span>
        ) : (
          <span>⚠ Upload doc first</span>
        )}
      </div>

      {/* Configuration Warning */}
      {configWarning && (
        <div className="demo-warning">
          ⚠ {configWarning}
        </div>
      )}

      {/* Demo Buttons */}
      <div className="demo-buttons">
        <button
          onClick={() => onRunDemo(DEMO_Q1)}
          disabled={disabled || !isDemoReady || !DEMO_Q1}
          className="demo-button demo-grounded"
          title={!DEMO_Q1 ? 'VITE_DEMO_Q1 not configured' : undefined}
        >
          Run Grounded Demo
        </button>
        <button
          onClick={() => onRunDemo(DEMO_Q2)}
          disabled={disabled || !isDemoReady || !DEMO_Q2}
          className="demo-button demo-refusal"
          title={!DEMO_Q2 ? 'VITE_DEMO_Q2 not configured' : undefined}
        >
          Run Refusal Demo
        </button>
        <button
          onClick={onClear}
          disabled={disabled}
          className="demo-button demo-clear"
        >
          Clear Conversation
        </button>
      </div>
    </div>
  );
}
