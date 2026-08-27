import { useState } from 'react';
import { Link } from 'react-router-dom';
import './Demo.css';

const steps = [
  {
    id: 1,
    title: '📁 Document Indexing',
    subtitle: 'Upload and automatic processing',
    content: (
      <>
        <p className="demo-script">
          "First, let's see how easy it is to get started. Simply upload your documents—PDFs, 
          Word files, or text documents. RAGify automatically indexes them into a vector database. 
          You can see the status here: documents go from 'pending' to 'indexed' within seconds."
        </p>
        <div className="demo-action">
          <Link to="/docs" className="demo-link-button">
            Go to Documents →
          </Link>
        </div>
        <ul className="demo-points">
          <li>No manual data entry required</li>
          <li>Automatic chunking and embedding</li>
          <li>Real-time status tracking</li>
        </ul>
      </>
    ),
  },
  {
    id: 2,
    title: '✅ Grounded Question',
    subtitle: 'Ask about content in your documents',
    content: (
      <>
        <p className="demo-script">
          "Now let's ask a question about information that's actually in the documents. 
          Watch how RAGify retrieves the most relevant evidence and generates a grounded 
          answer based solely on your data."
        </p>
        <div className="demo-action">
          <Link to="/query" className="demo-link-button">
            Go to Query →
          </Link>
        </div>
        <div className="demo-examples">
          <strong>Example questions:</strong>
          <ul>
            <li>"What is the vacation policy?"</li>
            <li>"What are the security requirements for contractors?"</li>
            <li>"How do I submit an expense report?"</li>
          </ul>
        </div>
        <ul className="demo-points">
          <li>Answers grounded in your documents</li>
          <li>No hallucinations—only factual responses</li>
          <li>Fast retrieval (&lt;2 seconds typical)</li>
        </ul>
      </>
    ),
  },
  {
    id: 3,
    title: '🔍 Evidence Transparency',
    subtitle: 'Show source snippets and citations',
    content: (
      <>
        <p className="demo-script">
          "Here's what makes RAGify trustworthy: every answer comes with evidence. 
          You can expand these cards to see the exact text snippets used, with your query 
          terms highlighted. This is full transparency—you can verify every claim."
        </p>
        <ul className="demo-points">
          <li>Collapsible evidence cards</li>
          <li>Highlighted matching terms</li>
          <li>Source filenames and chunk IDs</li>
          <li>Copy snippets to clipboard</li>
        </ul>
        <div className="demo-tip">
          <strong>💡 Pro tip:</strong> Enable the debug drawer (Ctrl+D) to show retrieval 
          scores and backend details.
        </div>
      </>
    ),
  },
  {
    id: 4,
    title: '🚫 Refusal Demo',
    subtitle: 'Out-of-scope question handling',
    content: (
      <>
        <p className="demo-script">
          "But what happens if you ask about something that's not in your documents? 
          RAGify refuses to answer rather than making something up. This is the 'grounding 
          gate'—a safety mechanism that prevents hallucinations."
        </p>
        <div className="demo-examples">
          <strong>Example questions:</strong>
          <ul>
            <li>"What's the weather today?"</li>
            <li>"Who won the Super Bowl last year?"</li>
            <li>"Tell me about quantum computing."</li>
          </ul>
        </div>
        <ul className="demo-points">
          <li>Clear refusal banner</li>
          <li>Explanation of why it couldn't answer</li>
          <li>No evidence shown (because there is none)</li>
          <li>Prevents misinformation</li>
        </ul>
      </>
    ),
  },
  {
    id: 5,
    title: '🔐 Privacy & Deployment',
    subtitle: 'Enterprise-ready architecture',
    content: (
      <>
        <p className="demo-script">
          "Let's talk about privacy and deployment. RAGify is designed for enterprise environments:"
        </p>
        
        <div className="demo-subsection">
          <h4>Privacy</h4>
          <ul>
            <li>Your data never leaves your infrastructure</li>
            <li>Self-hosted deployment options (Azure, AWS, on-prem)</li>
            <li>No data sent to third-party APIs</li>
            <li>Full GDPR/SOC2 compliance support</li>
          </ul>
        </div>

        <div className="demo-subsection">
          <h4>Deployment Options</h4>
          <ul>
            <li><strong>Cloud:</strong> Deploy to your Azure subscription or AWS account</li>
            <li><strong>On-Premises:</strong> Run on your own hardware</li>
            <li><strong>Hybrid:</strong> Index on-prem, query from cloud</li>
          </ul>
        </div>

        <div className="demo-subsection">
          <h4>Multi-tenancy</h4>
          <ul>
            <li>Isolated collections per tenant</li>
            <li>Role-based access control</li>
            <li>Audit logs for compliance</li>
          </ul>
        </div>

        <p className="demo-script">
          "This means you get the power of AI without compromising on security or control."
        </p>
      </>
    ),
  },
];

export default function Demo() {
  const [activeStep, setActiveStep] = useState(1);
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set([1]));

  const toggleStep = (stepId: number) => {
    const newExpanded = new Set(expandedSteps);
    if (newExpanded.has(stepId)) {
      newExpanded.delete(stepId);
    } else {
      newExpanded.add(stepId);
    }
    setExpandedSteps(newExpanded);
    setActiveStep(stepId);
  };

  const expandAll = () => {
    setExpandedSteps(new Set(steps.map(s => s.id)));
  };

  const collapseAll = () => {
    setExpandedSteps(new Set());
  };

  return (
    <div className="demo-page">
      <div className="demo-container">
        <header className="demo-header">
          <h1>🎯 RAGify Demo Script</h1>
          <p className="demo-subtitle">
            5-step sales demonstration flow • ~6 minutes + Q&A
          </p>
          <div className="demo-controls">
            <button onClick={expandAll} className="demo-control-btn">
              Expand All
            </button>
            <button onClick={collapseAll} className="demo-control-btn">
              Collapse All
            </button>
          </div>
        </header>

        <div className="demo-steps">
          {steps.map((step) => {
            const isExpanded = expandedSteps.has(step.id);
            const isActive = activeStep === step.id;

            return (
              <div
                key={step.id}
                className={`demo-step ${isExpanded ? 'expanded' : ''} ${isActive ? 'active' : ''}`}
              >
                <div
                  className="demo-step-header"
                  onClick={() => toggleStep(step.id)}
                >
                  <div className="demo-step-title">
                    <span className="demo-step-number">Step {step.id}</span>
                    <div>
                      <h3>{step.title}</h3>
                      <p className="demo-step-subtitle">{step.subtitle}</p>
                    </div>
                  </div>
                  <button className="demo-step-toggle">
                    {isExpanded ? '−' : '+'}
                  </button>
                </div>

                {isExpanded && (
                  <div className="demo-step-content">
                    {step.content}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <footer className="demo-footer">
          <div className="demo-closing">
            <h3>Closing</h3>
            <p className="demo-script">
              "So that's RAGify: secure, transparent, and grounded in your documents. 
              No hallucinations, full evidence, and complete control over your data. 
              Ready to get started?"
            </p>
            <div className="demo-next-steps">
              <h4>Next Steps</h4>
              <ul>
                <li>Schedule technical deep-dive</li>
                <li>Provide trial access</li>
                <li>Share deployment guide</li>
                <li>Discuss custom integrations</li>
              </ul>
            </div>
          </div>

          <div className="demo-faq">
            <h3>Common Questions</h3>
            <dl>
              <dt>How accurate is the retrieval?</dt>
              <dd>Vector similarity typically achieves 90%+ recall on well-structured documents. 
                  You can tune top_k and chunking strategies.</dd>

              <dt>What document formats are supported?</dt>
              <dd>PDF, DOCX, TXT, and more. We extract text and preserve structure.</dd>

              <dt>Can it handle multiple languages?</dt>
              <dd>Yes! The embedding models support 100+ languages.</dd>

              <dt>What's the cost?</dt>
              <dd>Depends on deployment model. Self-hosted costs are primarily compute. 
                  Cloud pricing varies by provider.</dd>
            </dl>
          </div>
        </footer>
      </div>
    </div>
  );
}
