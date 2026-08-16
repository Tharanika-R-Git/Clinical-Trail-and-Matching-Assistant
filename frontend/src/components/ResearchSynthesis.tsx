import { useState } from 'react'

interface ReportData {
  patient_id: string
  trial_id: string
  report_text: string
  report_sections: {
    patient_profile: Record<string, unknown>
    trial_matching: { trial_id: string; eligibility_status: string | null; match_score: number | null }
    clinical_changes: Record<string, { baseline: number | null; followup: number | null; change: number | null; unit: string }>
    endpoint_evaluation: { primary: { endpoint_name: string; result: string }; secondary: Array<{ endpoint_name: string; result: string }> }
    safety_summary: { total_count: number; serious_count: number; summary: string }
    ml_prediction: { model_name: string; prediction: number | null; probability: number | null }
    rag_evidence: Array<{ text: string; score: number; source: string }>
    shap_contributions: Array<{ feature: string; contribution: number }>
  }
  generated_at: string
  source: string
}

interface Props {
  patientId: string | null
  matchScore: number | null
  eligibilityStatus: string | null
}

export default function ResearchSynthesis({ patientId, matchScore, eligibilityStatus }: Props) {
  const [report, setReport] = useState<ReportData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  const generateReport = async () => {
    if (!patientId) return
    setLoading(true)
    setError(null)
    try {
      const r = await fetch('/api/research/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: patientId,
          trial_id: 'NCT05502562',
          match_score: matchScore,
          eligibility_status: eligibilityStatus,
          include_rag_evidence: true,
          rag_top_k: 5,
        })
      })
      if (!r.ok) throw new Error(`API error ${r.status}`)
      const data = await r.json()
      setReport(data)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const copyReport = () => {
    if (report?.report_text) {
      navigator.clipboard.writeText(report.report_text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  if (!patientId) return (
    <div className="view-container">
      <div className="view-header"><h1 className="view-title">Research Synthesis</h1></div>
      <div className="empty-state">
        <div className="empty-icon">📋</div>
        <div className="empty-title">No Patient Selected</div>
        <div className="empty-sub">Select a matched patient and generate a complete auditable research report with evidence citations.</div>
      </div>
    </div>
  )

  return (
    <div className="view-container">
      <div className="view-header">
        <h1 className="view-title">Research Synthesis</h1>
        <div className="header-actions">
          <div className="patient-pill">Patient: {patientId}</div>
          <button id="generate-report-btn" className="btn-primary" onClick={generateReport} disabled={loading}>
            {loading ? <><span className="spinner-sm" /> Generating...</> : '📋 Generate Report'}
          </button>
          {report && (
            <button id="copy-report-btn" className="btn-secondary" onClick={copyReport}>
              {copied ? '✓ Copied!' : '⎘ Copy'}
            </button>
          )}
        </div>
      </div>

      {error && <div className="error-banner">⚠ {error}</div>}

      {!report && !loading && (
        <div className="info-card">
          <span className="info-icon">ℹ</span>
          Click "Generate Report" to synthesise a full auditable research report combining eligibility, clinical changes, endpoint evaluation, ML prediction, SHAP explanations, and RAG evidence citations.
        </div>
      )}

      {report && (
        <>
          {/* Summary Cards */}
          <div className="report-summary-grid">
            <div className="summary-card">
              <div className="sum-label">Match Score</div>
              <div className="sum-val">{report.report_sections.trial_matching.match_score?.toFixed(1) ?? 'N/A'}</div>
            </div>
            <div className="summary-card">
              <div className="sum-label">ML Risk</div>
              <div className="sum-val">{report.report_sections.ml_prediction.probability != null
                ? `${(report.report_sections.ml_prediction.probability * 100).toFixed(0)}%` : 'N/A'}
              </div>
            </div>
            <div className="summary-card">
              <div className="sum-label">Primary Endpoint</div>
              <div className={`sum-val ${report.report_sections.endpoint_evaluation.primary.result === 'ACHIEVED' ? 'sum-green' : 'sum-red'}`}>
                {report.report_sections.endpoint_evaluation.primary.result}
              </div>
            </div>
            <div className="summary-card">
              <div className="sum-label">Adverse Events</div>
              <div className="sum-val">{report.report_sections.safety_summary.total_count}</div>
            </div>
          </div>

          {/* RAG Evidence */}
          {report.report_sections.rag_evidence.length > 0 && (
            <div className="card">
              <div className="card-header">📚 Evidence Citations ({report.report_sections.rag_evidence.length})</div>
              <div className="evidence-list">
                {report.report_sections.rag_evidence.map((ev, i) => (
                  <div key={i} className="evidence-item">
                    <div className="evidence-rank">[{i + 1}]</div>
                    <div className="evidence-content">
                      <div className="evidence-text">{ev.text}</div>
                      <div className="evidence-meta">
                        <span className="evidence-source">{ev.source}</span>
                        <span className="evidence-score">Score: {ev.score.toFixed(4)}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Full Report Text */}
          <div className="card">
            <div className="card-header">
              Full Auditable Report
              <span className="report-meta">Generated: {new Date(report.generated_at).toLocaleString()}</span>
            </div>
            <pre className="report-text" id="report-full-text">{report.report_text}</pre>
          </div>

          <div className="disclaimer-card">
            <span className="disclaimer-icon">🔒</span>
            {report.source}. All clinical conclusions require independent medical validation.
          </div>
        </>
      )}
    </div>
  )
}
