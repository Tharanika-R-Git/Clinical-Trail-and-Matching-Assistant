import { useEffect, useState } from 'react'

interface TrialData {
  trial_id: string
  title: string
  condition: string
  phase: string
  intervention: { name: string; type: string; mechanism?: string; doses?: string[] }
  eligibility: { inclusion: string[]; exclusion: string[] }
  outcomes: { primary: string[]; secondary: string[] }
  study_design: Record<string, string>
  timeframes: string[]
}

export default function TrialView() {
  const [trial, setTrial] = useState<TrialData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/trials/NCT05502562')
      .then(r => r.json())
      .then(data => { setTrial(data); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return <div className="loading-state"><div className="spinner" />Loading trial data...</div>
  if (error) return <div className="error-state">⚠ {error}</div>
  if (!trial) return null

  return (
    <div className="view-container">
      <div className="view-header">
        <h1 className="view-title">Trial Profile</h1>
        <div className="trial-id-badge">{trial.trial_id}</div>
      </div>

      {/* Trial Hero */}
      <div className="card trial-hero">
        <div className="trial-hero-content">
          <h2 className="trial-title">{trial.title || 'PIONEER PLUS Study'}</h2>
          <div className="trial-meta-row">
            <span className="meta-pill">{trial.phase}</span>
            <span className="meta-pill meta-condition">{trial.condition}</span>
            <span className="meta-pill meta-drug">{trial.intervention.name}</span>
          </div>
          <div className="trial-mechanism">
            Mechanism: {trial.intervention.mechanism || 'GLP-1 Receptor Agonist'}
          </div>
        </div>
      </div>

      {/* Timeframes */}
      <div className="card">
        <div className="card-header">Study Timeframes</div>
        <div className="timeframes-track">
          {trial.timeframes.map((tf, i) => (
            <div key={tf} className="timeframe-node">
              <div className="timeframe-dot" />
              <div className="timeframe-label">{tf}</div>
              {i < trial.timeframes.length - 1 && <div className="timeframe-connector" />}
            </div>
          ))}
        </div>
      </div>

      <div className="two-col-grid">
        {/* Eligibility */}
        <div className="card">
          <div className="card-header card-header-green">✓ Inclusion Criteria</div>
          <ul className="criteria-list">
            {trial.eligibility.inclusion.map((c, i) => (
              <li key={i} className="criteria-item criteria-include">
                <span className="criteria-icon">✓</span>
                {c}
              </li>
            ))}
          </ul>
        </div>
        <div className="card">
          <div className="card-header card-header-red">✗ Exclusion Criteria</div>
          <ul className="criteria-list">
            {trial.eligibility.exclusion.map((c, i) => (
              <li key={i} className="criteria-item criteria-exclude">
                <span className="criteria-icon">✗</span>
                {c}
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Outcomes */}
      <div className="card">
        <div className="card-header">Clinical Endpoints</div>
        <div className="endpoints-grid">
          <div className="endpoint-group">
            <div className="endpoint-type primary">PRIMARY</div>
            {trial.outcomes.primary.map((ep, i) => (
              <div key={i} className="endpoint-item endpoint-primary">{ep}</div>
            ))}
          </div>
          <div className="endpoint-group">
            <div className="endpoint-type secondary">SECONDARY</div>
            {trial.outcomes.secondary.map((ep, i) => (
              <div key={i} className="endpoint-item endpoint-secondary">{ep}</div>
            ))}
          </div>
        </div>
      </div>

      {/* Study Design */}
      <div className="card">
        <div className="card-header">Study Design</div>
        <div className="design-grid">
          {Object.entries(trial.study_design).filter(([k]) => k !== 'enrollment').map(([key, val]) => (
            <div key={key} className="design-item">
              <div className="design-key">{key.replace(/_/g, ' ')}</div>
              <div className="design-val">{val}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
