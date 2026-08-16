import { useState } from 'react'

interface Candidate {
  patient_id: string
  match_score: number
  eligibility_status: string
  passed: number
  failed: number
  unknown: number
  manual_review: number
  demographics: { age: number | null; gender: string | null; location: string | null }
  clinical_metrics: { hba1c: number | null; fasting_glucose: number | null; weight: number | null; egfr: number | null; current_medication: string | null }
}

interface MatchResult {
  best_patient_id: string
  match_score: number
  status: string
  candidates: Candidate[]
}

interface Props {
  onPatientSelect: (p: { patient_id: string; match_score: number; eligibility_status: string }) => void
}

const statusColor = (s: string) => {
  if (s === 'POTENTIALLY_ELIGIBLE') return 'badge-green'
  if (s === 'POTENTIALLY_ELIGIBLE_WITH_REVIEW') return 'badge-yellow'
  return 'badge-red'
}
const statusLabel = (s: string) => {
  if (s === 'POTENTIALLY_ELIGIBLE') return 'Eligible'
  if (s === 'POTENTIALLY_ELIGIBLE_WITH_REVIEW') return 'Review'
  return 'Ineligible'
}

export default function MatchingRanker({ onPatientSelect }: Props) {
  const [result, setResult] = useState<MatchResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const runMatching = async () => {
    setLoading(true)
    setError(null)
    try {
      const r = await fetch('/api/matching/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trial_id: 'NCT05502562' })
      })
      if (!r.ok) throw new Error(`API error ${r.status}`)
      const data: MatchResult = await r.json()
      setResult(data)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const handleSelect = (c: Candidate) => {
    setSelectedId(c.patient_id)
    onPatientSelect({ patient_id: c.patient_id, match_score: c.match_score, eligibility_status: c.eligibility_status })
  }

  return (
    <div className="view-container">
      <div className="view-header">
        <h1 className="view-title">Patient–Trial Matching</h1>
        <button id="run-matching-btn" className="btn-primary" onClick={runMatching} disabled={loading}>
          {loading ? <><span className="spinner-sm" /> Running...</> : '⚡ Run Matching'}
        </button>
      </div>

      {error && <div className="error-banner">⚠ {error}</div>}

      {!result && !loading && (
        <div className="empty-state">
          <div className="empty-icon">🎯</div>
          <div className="empty-title">Start Matching</div>
          <div className="empty-sub">Click "Run Matching" to rank volunteers against NCT05502562 eligibility criteria using BM25 + Dense retrieval.</div>
        </div>
      )}

      {result && (
        <>
          {/* Best Match Hero */}
          <div className="card best-match-hero">
            <div className="best-match-label">🏆 Best Matched Patient</div>
            <div className="best-match-id">{result.best_patient_id}</div>
            <div className="best-match-row">
              <div className="stat-block">
                <div className="stat-num">{result.match_score.toFixed(1)}</div>
                <div className="stat-label">Match Score</div>
              </div>
              <div className={`badge ${statusColor(result.status)}`}>{statusLabel(result.status)}</div>
            </div>
          </div>

          {/* Candidate Table */}
          <div className="card">
            <div className="card-header">Ranked Candidates ({result.candidates.length})</div>
            <div className="table-wrap">
              <table className="data-table" id="candidates-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Patient ID</th>
                    <th>Score</th>
                    <th>Status</th>
                    <th>✓ Pass</th>
                    <th>✗ Fail</th>
                    <th>? Unknown</th>
                    <th>Age</th>
                    <th>HbA1c</th>
                    <th>eGFR</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {result.candidates.map((c, i) => (
                    <tr
                      key={c.patient_id}
                      className={`table-row ${selectedId === c.patient_id ? 'row-selected' : ''} ${c.failed > 0 ? 'row-ineligible' : ''}`}
                    >
                      <td className="rank-col">{i + 1}</td>
                      <td className="id-col">{c.patient_id}</td>
                      <td>
                        <div className="score-bar-wrap">
                          <div className="score-bar" style={{ width: `${Math.min(c.match_score, 100)}%` }} />
                          <span className="score-val">{c.match_score.toFixed(1)}</span>
                        </div>
                      </td>
                      <td><span className={`badge ${statusColor(c.eligibility_status)}`}>{statusLabel(c.eligibility_status)}</span></td>
                      <td className="pass-col">{c.passed}</td>
                      <td className="fail-col">{c.failed}</td>
                      <td className="unknown-col">{c.unknown}</td>
                      <td>{c.demographics.age ?? '–'}</td>
                      <td>{c.clinical_metrics.hba1c != null ? `${c.clinical_metrics.hba1c}%` : '–'}</td>
                      <td>{c.clinical_metrics.egfr ?? '–'}</td>
                      <td>
                        <button
                          id={`select-patient-${c.patient_id}`}
                          className="btn-sm"
                          onClick={() => handleSelect(c)}
                        >
                          Analyse →
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
