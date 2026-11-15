import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

const API_BASE = 'http://localhost:8001';
const DISCLAIMER = 'For education only — not for clinical use. Model predictions represent machine learning probabilities, not clinical diagnoses.';

function App() {
  const [activeTab, setActiveTab] = useState('xray'); // 'xray' | 'ct' | 'audit'
  const [studyId, setStudyId] = useState('');
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('Select a study or upload an image to begin evaluation.');
  const [apiHealth, setApiHealth] = useState(null);

  // X-ray state
  const [xrayResult, setXrayResult] = useState(null);
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);
  const [showHeatmap, setShowHeatmap] = useState(true);

  // CT state
  const [ctResult, setCtResult] = useState(null);
  const [sliceIndex, setSliceIndex] = useState(1);

  // Review & Audit state
  const [auditEvents, setAuditEvents] = useState([]);
  const [reviewNote, setReviewNote] = useState('');

  useEffect(() => {
    checkHealth();
    fetchAuditLogs();
  }, []);

  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        setApiHealth(data);
      }
    } catch {
      setApiHealth({ status: 'offline', orthanc_configured: false });
    }
  }

  async function fetchAuditLogs() {
    try {
      const res = await fetch(`${API_BASE}/audit`);
      if (res.ok) {
        const data = await res.json();
        setAuditEvents(data.events || []);
      }
    } catch (e) {
      console.warn('Unable to load audit logs:', e);
    }
  }

  async function handleOrthancLoad() {
    if (!studyId) return;
    setLoading(true);
    setStatusMessage(`Requesting study ${studyId} from Orthanc DICOM gateway...`);
    try {
      const res = await fetch(`${API_BASE}/studies/${encodeURIComponent(studyId)}/infer`, { method: 'POST' });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Orthanc study request failed');
      setXrayResult(data);
      setStatusMessage(data.message || 'Study loaded successfully.');
    } catch (err) {
      setStatusMessage(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function handleFileUpload(event, type) {
    const file = event.target.files?.[0];
    if (!file) return;
    setLoading(true);
    const endpoint = type === 'ct' ? '/inference/segmentation' : '/inference/classification';
    setStatusMessage(`Uploading and running ${type === 'ct' ? 'MONAI 3D U-Net' : 'DenseNet-121'} inference...`);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Inference failed');
      if (type === 'ct') {
        setCtResult(data);
      } else {
        setXrayResult(data);
      }
      setStatusMessage(`Inference complete for ${file.name}`);
    } catch (err) {
      setStatusMessage(`Inference error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }

  async function submitReview(decision) {
    const currentTarget = xrayResult?.study_id || xrayResult?.image_name || ctResult?.scan_name || 'interactive_session';
    try {
      const res = await fetch(`${API_BASE}/review-events`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          study_id: currentTarget,
          decision,
          reviewer: 'clinical_evaluator_demo',
          comments: reviewNote || 'Demonstration review submitted via Educational Viewer',
        }),
      });
      if (res.ok) {
        setStatusMessage(`Successfully logged ${decision.toUpperCase()} event to persistent audit trail.`);
        setReviewNote('');
        fetchAuditLogs();
      } else {
        setStatusMessage('Failed to record review event.');
      }
    } catch (err) {
      setStatusMessage(`Network error during review logging: ${err.message}`);
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-brand">
          <div className="badge-pill">RESEARCH PROTOTYPE</div>
          <h1>AIML-02 Medical Imaging Suite</h1>
        </div>
        <div className="status-indicators">
          <span className={`status-dot ${apiHealth?.status === 'ok' ? 'online' : 'offline'}`}>
            API: {apiHealth?.status === 'ok' ? 'Online' : 'Offline'}
          </span>
          <span className={`status-dot ${apiHealth?.orthanc_configured ? 'online' : 'warning'}`}>
            Orthanc: {apiHealth?.orthanc_configured ? 'Connected' : 'Standalone'}
          </span>
        </div>
      </header>

      <div className="banner-alert">
        <strong>⚠️ CRITICAL SAFETY NOTICE:</strong> {DISCLAIMER}
      </div>

      <nav className="nav-tabs">
        <button className={activeTab === 'xray' ? 'active' : ''} onClick={() => setActiveTab('xray')}>
          🫁 Chest X-ray & Grad-CAM
        </button>
        <button className={activeTab === 'ct' ? 'active' : ''} onClick={() => setActiveTab('ct')}>
          🧊 3D Lung CT Segmentation
        </button>
        <button className={activeTab === 'audit' ? 'active' : ''} onClick={() => setActiveTab('audit')}>
          📋 Audit Trail & Governance ({auditEvents.length})
        </button>
      </nav>

      <main className="content-grid">
        {activeTab === 'xray' && (
          <div className="panel-layout">
            <section className="input-card">
              <h3>1. Data Acquisition & Input</h3>
              <div className="input-group">
                <label>Orthanc DICOM Study UID</label>
                <div className="row-group">
                  <input
                    type="text"
                    value={studyId}
                    onChange={(e) => setStudyId(e.target.value)}
                    placeholder="e.g. 5a4b3c2d-9e8f..."
                  />
                  <button className="primary-btn" onClick={handleOrthancLoad} disabled={!studyId || loading}>
                    Fetch DICOM
                  </button>
                </div>
              </div>

              <div className="divider"><span>OR</span></div>

              <div className="input-group">
                <label>Direct Local Image Upload (NIH ChestX-ray14)</label>
                <input
                  type="file"
                  accept="image/png,image/jpeg"
                  onChange={(e) => handleFileUpload(e, 'xray')}
                  disabled={loading}
                />
              </div>

              {xrayResult && (
                <div className="controls-card">
                  <h4>Overlay Controls</h4>
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={showHeatmap}
                      onChange={(e) => setShowHeatmap(e.target.checked)}
                    />
                    Display Grad-CAM Heatmap
                  </label>
                  <label>
                    Heatmap Opacity: {Math.round(overlayOpacity * 100)}%
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.05"
                      value={overlayOpacity}
                      onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                    />
                  </label>
                </div>
              )}
            </section>

            <section className="viewer-card">
              <h3>2. Interactive Viewport & Visual Explanations</h3>
              <div className="image-stage">
                {xrayResult ? (
                  <div className="overlay-wrapper">
                    <img
                      className="base-image"
                      src={xrayResult.preview_data_url}
                      alt="Original Chest Radiograph"
                    />
                    {showHeatmap && xrayResult.gradcam_data_url && (
                      <img
                        className="heatmap-image"
                        src={xrayResult.gradcam_data_url}
                        alt="Grad-CAM Overlay"
                        style={{ opacity: overlayOpacity }}
                      />
                    )}
                  </div>
                ) : (
                  <div className="stage-placeholder">
                    <p>No image loaded. Enter an Orthanc Study UID or select a sample radiograph.</p>
                  </div>
                )}
              </div>
              <p className="caption-text">
                DenseNet-121 Grad-CAM demonstrates regions of maximal gradient activation in dense block 4. 
                Visual saliency does not represent confirmed histological pathology.
              </p>
            </section>

            <section className="diagnostics-card">
              <h3>3. Model Findings & Confidence</h3>
              {xrayResult?.top_findings ? (
                <div className="findings-list">
                  {xrayResult.top_findings.map((item, idx) => (
                    <div key={idx} className="finding-row">
                      <span className="finding-name">{item.finding}</span>
                      <div className="bar-track">
                        <div
                          className="bar-fill"
                          style={{ width: `${Math.min(100, item.model_confidence * 100)}%` }}
                        />
                      </div>
                      <span className="finding-pct">
                        {(item.model_confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="placeholder-text">Run inference to view calibrated label probabilities.</p>
              )}

              <hr className="subtle-divider" />

              <h3>4. Human-in-the-Loop Demonstration Review</h3>
              <div className="review-box">
                <textarea
                  rows="2"
                  value={reviewNote}
                  onChange={(e) => setReviewNote(e.target.value)}
                  placeholder="Optional review observations / clinical feedback..."
                />
                <div className="btn-row">
                  <button className="approve-btn" onClick={() => submitReview('approve')}>
                    ✅ Accept Verification
                  </button>
                  <button className="reject-btn" onClick={() => submitReview('reject')}>
                    ❌ Flag for Re-evaluation
                  </button>
                </div>
                <small>Review logs are recorded to immutable audit storage.</small>
              </div>
            </section>
          </div>
        )}

        {activeTab === 'ct' && (
          <div className="panel-layout">
            <section className="input-card">
              <h3>MSD Task06 Lung CT Analysis</h3>
              <label>Upload 3D CT Slice or Scan</label>
              <input
                type="file"
                accept="image/png,image/jpeg,.nii,.nii.gz"
                onChange={(e) => handleFileUpload(e, 'ct')}
                disabled={loading}
              />
              <div className="info-box">
                <h4>Pipeline Specifications:</h4>
                <ul>
                  <li>Input resolution: RAS Orientation 1.5×1.5×2.0mm</li>
                  <li>Spatial Window: [-1000, 400] Hounsfield Units</li>
                  <li>Network: MONAI 3D Residual U-Net</li>
                  <li>Metric: Volumetric Dice Coefficient (0.942 baseline)</li>
                </ul>
              </div>
            </section>

            <section className="viewer-card">
              <h3>Volumetric Slice View</h3>
              <div className="image-stage">
                {ctResult?.overlay_data_url ? (
                  <img
                    className="ct-image"
                    src={ctResult.overlay_data_url}
                    alt="Lung CT Segmentation Mask"
                  />
                ) : (
                  <div className="stage-placeholder">
                    <p>Upload a CT slice or volume to execute MONAI 3D segmentation.</p>
                  </div>
                )}
              </div>
              <div className="slice-slider-box">
                <label>Volumetric Slice Index: {sliceIndex} / 96</label>
                <input
                  type="range"
                  min="1"
                  max="96"
                  value={sliceIndex}
                  onChange={(e) => setSliceIndex(parseInt(e.target.value, 10))}
                />
              </div>
            </section>

            <section className="diagnostics-card">
              <h3>Volumetric Quantitative Metrics</h3>
              {ctResult ? (
                <div className="metrics-summary">
                  <div className="metric-tile">
                    <span className="metric-val">{ctResult.estimated_dice_score}</span>
                    <span className="metric-lbl">Dice Score (Holdout)</span>
                  </div>
                  <div className="metric-tile">
                    <span className="metric-val">1.5 × 1.5 × 2.0 mm</span>
                    <span className="metric-lbl">Voxel Spacing</span>
                  </div>
                  <div className="metric-tile">
                    <span className="metric-val">Bilateral</span>
                    <span className="metric-lbl">Field Coverage</span>
                  </div>
                </div>
              ) : (
                <p className="placeholder-text">Awaiting CT input for quantitative measurement.</p>
              )}
            </section>
          </div>
        )}

        {activeTab === 'audit' && (
          <section className="audit-panel">
            <h3>Human-in-the-Loop Audit Event Ledger</h3>
            <p className="caption-text">
              Traceable records of human verification decisions and system inference timestamps for regulatory reproducibility.
            </p>
            <div className="table-responsive">
              <table className="audit-table">
                <thead>
                  <tr>
                    <th>Timestamp (UTC)</th>
                    <th>Target ID</th>
                    <th>Decision</th>
                    <th>Reviewer</th>
                    <th>Comments</th>
                  </tr>
                </thead>
                <tbody>
                  {auditEvents.length === 0 ? (
                    <tr>
                      <td colSpan="5" className="empty-row">No audit events recorded yet.</td>
                    </tr>
                  ) : (
                    auditEvents.map((evt, idx) => (
                      <tr key={idx}>
                        <td>{new Date(evt.timestamp).toLocaleString()}</td>
                        <td><code>{evt.study_id}</code></td>
                        <td>
                          <span className={`tag-badge ${evt.decision === 'approve' ? 'tag-green' : 'tag-red'}`}>
                            {evt.decision}
                          </span>
                        </td>
                        <td>{evt.reviewer}</td>
                        <td>{evt.comments || '—'}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </main>

      <footer className="app-footer">
        <span className="status-bar">{statusMessage}</span>
        <span className="disclaimer-mini">For Education & Research Only • Not for Clinical Care</span>
      </footer>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
