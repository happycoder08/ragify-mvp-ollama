import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { listDocuments, uploadDocuments, purgeDocuments } from '../api';
import { consumeDemoTokenFromUrl, getDemoToken } from '../utils/demoToken';
import type { DocumentRecord } from '../contracts/types';
import { getSelectedDocIds, toggleDocId, clearSelection } from '../utils/documentSelection';
import './Docs.css';

export default function Docs() {
  const navigate = useNavigate();

  const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === 'true';

  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [fetching, setFetching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [purgeLoading, setPurgeLoading] = useState(false);
  const [purgeError, setPurgeError] = useState<string | null>(null);
  const [purgeSuccess, setPurgeSuccess] = useState<string | null>(null);

  const [dragActive, setDragActive] = useState(false);
  const [pollingTimeoutReached, setPollingTimeoutReached] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);

  const [selectedDocs, setSelectedDocs] = useState<number[]>(() => getSelectedDocIds());
  const [demoToken, setDemoToken] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollingIntervalRef = useRef<number | null>(null);
  const pollingStartTimeRef = useRef<number | null>(null);

  // Fetch documents
  const fetchDocuments = async (showLoading = true) => {
    if (showLoading) setLoading(true);
    setFetching(true);
    setError(null);

    try {
      const docs = await listDocuments();
      // listDocuments may return { documents: [...] } or [...] depending on your api.ts
      const normalized = Array.isArray(docs) ? docs : (docs as any)?.documents ?? [];
      setDocuments(normalized);
      setLastRefreshed(new Date());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch documents');
    } finally {
      if (showLoading) setLoading(false);
      setFetching(false);
    }
  };

  const handleRefresh = () => {
    fetchDocuments(false);
  };

  // Purge indexed documents
  const handlePurge = async () => {
    setPurgeLoading(true);
    setPurgeError(null);
    setPurgeSuccess(null);
    try {
      const result = await purgeDocuments();
      setPurgeSuccess(result?.message || 'Indexed documents cleared');
      await fetchDocuments(false);
      setTimeout(() => setPurgeSuccess(null), 4000);
    } catch (err) {
      setPurgeError(err instanceof Error ? err.message : 'Failed to clear documents');
    } finally {
      setPurgeLoading(false);
    }
  };

  // Check if polling is needed
  const shouldPoll = (docs: DocumentRecord[]): boolean => {
    return docs.some(doc => doc.status === 'pending');
  };

  // Setup or clear polling based on document status
  useEffect(() => {
    const setupPolling = () => {
      // Clear existing interval
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }

      // Check if we need to poll
      if (shouldPoll(documents)) {
        // Start polling timer if not started
        if (!pollingStartTimeRef.current) {
          pollingStartTimeRef.current = Date.now();
          setPollingTimeoutReached(false);
        }

        // Check if 60 seconds have elapsed
        const elapsed = Date.now() - (pollingStartTimeRef.current || 0);
        if (elapsed >= 60000) {
          setPollingTimeoutReached(true);
          pollingStartTimeRef.current = null;
          return;
        }

        // Start polling every 2 seconds
        pollingIntervalRef.current = window.setInterval(() => {
          const elapsed = Date.now() - (pollingStartTimeRef.current || 0);
          if (elapsed >= 60000) {
            if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current);
              pollingIntervalRef.current = null;
            }
            setPollingTimeoutReached(true);
            pollingStartTimeRef.current = null;
          } else {
            fetchDocuments(false);
          }
        }, 2000);
      } else {
        // No polling needed, reset timer
        pollingStartTimeRef.current = null;
        setPollingTimeoutReached(false);
      }
    };

    setupPolling();

    // Cleanup on unmount
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [documents]);

  // Initial fetch
  useEffect(() => {
    // Consume demo token from URL (if present) on initial mount of /docs
    try {
      const t = consumeDemoTokenFromUrl();
      setDemoToken(t ?? getDemoToken());
    } catch {
      setDemoToken(getDemoToken());
    }

    fetchDocuments(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Handle file upload
  const handleUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return;

    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      const fileArray = Array.from(files);
      const result = await uploadDocuments(fileArray);

      const uploadedCount = (result as any)?.files_processed || 0;
      setUploadSuccess(`Successfully uploaded ${uploadedCount} file${uploadedCount !== 1 ? 's' : ''}`);

      // Immediately re-fetch documents
      await fetchDocuments(false);

      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }

      // Clear success message after 5 seconds
      setTimeout(() => setUploadSuccess(null), 5000);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  // Drag and drop handlers
  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleUpload(e.dataTransfer.files);
    }
  };

  // Handle document selection toggle
  const handleToggleDoc = (docId: number) => {
    const newSelection = toggleDocId(docId);
    setSelectedDocs(newSelection);
  };

  // Handle select/deselect all
  const handleToggleAll = () => {
    const indexedDocs = documents.filter(d => d.status === 'indexed');
    if (selectedDocs.length === indexedDocs.length && indexedDocs.length > 0) {
      // Deselect all
      clearSelection();
      setSelectedDocs([]);
    } else {
      // Select all indexed
      const allIndexedIds = indexedDocs.map(d => d.id);
      setSelectedDocs(allIndexedIds);
      import('../utils/documentSelection').then(({ setSelectedDocIds }) => {
        setSelectedDocIds(allIndexedIds);
      });
    }
  };

  // Determine banner message
  const isPolling = shouldPoll(documents);
  const hasIndexedDocs = documents.some(doc => doc.status === 'indexed');
  const isActive = uploading || fetching;

  let bannerMessage = '';
  let bannerClass = '';

  if (isPolling) {
    bannerMessage = pollingTimeoutReached
      ? 'Indexing is taking longer than expected. Documents may still be processing.'
      : 'Indexing documents...';
    bannerClass = 'banner-indexing';
  } else if (hasIndexedDocs) {
    bannerMessage = '✓ Ready to query';
    bannerClass = 'banner-ready';
  }

  // Format date
  const formatDate = (dateStr: string) => {
    try {
      return new Date(dateStr).toLocaleString();
    } catch {
      return dateStr;
    }
  };

  return (
    <div className="docs-page">
      <div className="docs-content">
        <div className="page-guidance">
          <div className="step-header">
            <span className="step-badge">Step 1 of 2</span>
            <h1>Upload Documents</h1>
            {(DEMO_MODE || demoToken) && (
              <span
                className="demo-access-badge"
                title="This is demo access, not real authentication."
              >
                {demoToken ? 'Demo access link' : 'Demo mode'}
              </span>
            )}
          </div>
          <p className="step-description">
            Upload your documents to build your knowledge base. Once indexed, you can ask questions in the Chat.
          </p>
        </div>

        <div className="docs-grid">
          {/* Left Column: Upload & Status */}
          <div className="left-column">
            {/* Status Banner */}
            {bannerMessage && (
              <div className={`status-card ${bannerClass}`}>
                <div className="status-content">
                  {isPolling && <span className="spinner small"></span>}
                  <span>{bannerMessage}</span>
                </div>
              </div>
            )}

            {/* Upload Card */}
            <div className="upload-card">
              <h2>Upload Files</h2>
              <div
                className={`upload-area ${dragActive ? 'drag-active' : ''} ${isActive ? 'disabled' : ''}`}
                onDragEnter={handleDragEnter}
                onDragLeave={handleDragLeave}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  onChange={(e) => handleUpload(e.target.files)}
                  className="file-input"
                  id="file-upload"
                  disabled={isActive}
                />
                <label htmlFor="file-upload" className="upload-label">
                  {uploading ? (
                    <>
                      <span className="spinner"></span>
                      <span>Uploading...</span>
                    </>
                  ) : fetching ? (
                    <>
                      <span className="spinner"></span>
                      <span>Refreshing...</span>
                    </>
                  ) : (
                    <>
                      <span className="upload-icon">📁</span>
                      <span>Click to select files or drag & drop</span>
                      <span className="upload-hint">PDF, TXT, DOC, DOCX</span>
                    </>
                  )}
                </label>
              </div>
              {uploadSuccess && <div className="success-message">{uploadSuccess}</div>}
              {uploadError && <div className="error-message">{uploadError}</div>}
            </div>


            {/* Action Buttons */}
            <div className="action-buttons">
              <button
                className="btn-primary"
                onClick={() => navigate('/query')}
                disabled={!hasIndexedDocs}
                title={!hasIndexedDocs ? 'Upload and index at least one document first' : 'Go to chat'}
              >
                {hasIndexedDocs ? '→ Go to Chat (Step 2)' : '→ Go to Chat'}
              </button>
              <button
                className="btn-secondary"
                onClick={() => fileInputRef.current?.click()}
                disabled={isActive}
              >
                + Upload Another
              </button>
            </div>
          </div>

          {/* Right Column: Documents Table */}
          <div className="right-column">
            <div className="table-card">
              <div className="table-header">
                <div className="table-title-row">
                  <h2>Your Documents</h2>
                  <div className="table-actions">
                    {documents.some(d => d.status === 'indexed') && (
                      <span className="active-docs-indicator">
                        Active: {selectedDocs.length === 0 ? 'All' : selectedDocs.length}
                      </span>
                    )}
                    <button
                      className="refresh-button"
                      onClick={handleRefresh}
                      disabled={fetching}
                      title="Refresh documents"
                    >
                      <span className={`refresh-icon ${fetching ? 'spinning' : ''}`}>↻</span>
                    </button>
                  </div>
                </div>
                {lastRefreshed && (
                  <div className="last-refreshed">
                    Last refreshed: {lastRefreshed.toLocaleTimeString('en-US', {
                      hour: '2-digit',
                      minute: '2-digit',
                      second: '2-digit',
                      hour12: false
                    })}
                  </div>
                )}
              </div>

              {loading && documents.length === 0 ? (
                <div className="loading">
                  <span className="spinner"></span>
                  <span>Loading documents...</span>
                </div>
              ) : error ? (
                <div className="error-message">{error}</div>
              ) : documents.length === 0 ? (
                <div className="empty-state">
                  <span className="empty-icon">📄</span>
                  <p>No documents yet</p>
                  <p className="empty-hint">Upload your first document to get started</p>
                </div>
              ) : (
                <div className={`table-container ${fetching ? 'refreshing' : ''}`}>
                  <table className="documents-table">
                    <thead>
                      <tr>
                        <th className="checkbox-col">
                          <input
                            type="checkbox"
                            checked={
                              selectedDocs.length > 0 &&
                              selectedDocs.length === documents.filter(d => d.status === 'indexed').length
                            }
                            onChange={handleToggleAll}
                            disabled={documents.filter(d => d.status === 'indexed').length === 0}
                            title="Select/deselect all indexed documents"
                          />
                        </th>
                        <th>Filename</th>
                        <th>Status</th>
                        <th>Uploaded</th>
                        <th>Error</th>
                      </tr>
                    </thead>
                    <tbody>
                      {documents.map((doc) => {
                        const isIndexed = doc.status === 'indexed';
                        const isSelected = selectedDocs.includes(doc.id);
                        return (
                          <tr key={doc.id}>
                            <td className="checkbox-col">
                              <input
                                type="checkbox"
                                checked={isSelected}
                                onChange={() => handleToggleDoc(doc.id)}
                                disabled={!isIndexed}
                                title={isIndexed ? 'Include in chat queries' : 'Document must be indexed first'}
                              />
                            </td>
                            <td className="filename-cell">{doc.filename}</td>
                            <td>
                              <span className={`status-badge status-${doc.status}`}>
                                {doc.status === 'pending' && <span className="status-spinner"></span>}
                                {doc.status}
                              </span>
                            </td>
                            <td className="date-cell">
                              {doc.created_at ? formatDate(doc.created_at) : '-'}
                            </td>
                            <td className="error-cell">
                              {doc.error_message || '-'}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            {/* Clear indexed docs button at the bottom of the table grid */}
            <div style={{ marginTop: 20, textAlign: 'right' }}>
              <button
                className="btn-primary"
                onClick={handlePurge}
                disabled={purgeLoading || isActive || !hasIndexedDocs}
                title={!hasIndexedDocs ? 'No indexed documents to clear' : 'Clear all indexed documents'}
              >
                {purgeLoading ? 'Clearing...' : 'Clear indexed docs'}
              </button>
              {purgeError && <div className="error-message">{purgeError}</div>}
              {purgeSuccess && <div className="success-message">{purgeSuccess}</div>}
            </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
