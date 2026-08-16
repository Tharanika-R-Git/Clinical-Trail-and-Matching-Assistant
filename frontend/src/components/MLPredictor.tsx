import { useEffect, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'

interface MLResult {
  model_name: string
  prediction: number
  probability: number
  model_metrics: Record<string, { roc_auc: number; pr_auc: number; accuracy: number; f1: number }>
}

interface SHAPResult {
  feature_contributions: Array<{ feature: string; contribution: number; value?: string }>
}

interface EndpointResult {
  primary: { endpoint_name: string; result: string; observed_change: number | null; unit: string }
  secondary: Array<{ endpoint_name: string; result: string; observed_change: number | null; unit: string }>
}

interface Props { patientId: string | null }

const resultColor = (r: string) => {
  if (r === 'ACHIEVED') return 'var(--green)'
  if (r === 'NOT_ACHIEVED') return 'var(--red)'
  return 'var(--yellow)'
}

export default function MLPredictor({ patientId }: Props) {
  const [mlData, setMlData] = useState<MLResult | null>(null)
  const [shapData, setShapData] = useState<SHAPResult | null>(null)
  const [epData, setEpData] = useState<EndpointResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!patientId) return
    setLoading(true)
    setError(null)

    Promise.all([
      fetch('/api/analysis/ml-predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ patient_id: patientId }) }).then(r => r.json()),
      fetch('/api/analysis/shap-explain', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ patient_id: patientId }) }).then(r => r.json()),
      fetch('/api/analysis/endpoints', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ patient_id: patientId }) }).then(r => r.json()),
    ])
      .then(([ml, shap, ep]) => {
        setMlData(ml)
        setShapData(shap)
        setEpData(ep)
        setLoading(false)
      })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [patientId])

  if (!patientId) return (
    <div className="view-container">
      <div className="view-header"><h1 className="view-title">ML Predictor</h1></div>
      <div className="empty-state">
        <div className="empty-icon">🤖</div>
        <div className="empty-title">No Patient Selected</div>
        <div className="empty-sub">Select a matched patient to run ML outcome prediction and SHAP analysis.</div>
      </div>
    </div>
  )

  if (loading) return <div className="loading-state"><div className="spinner" />Running ML pipeline...</div>
  if (error) return <div className="error-state">⚠ {error}</div>

  const shapChartData = shapData?.feature_contributions
    .slice(0, 8)
    .map(f => ({ feature: f.feature.replace('_', ' '), contribution: f.contribution, abs: Math.abs(f.contribution) }))
    .sort((a, b) => b.abs - a.abs) || []

  const riskLevel = mlData ? (mlData.probability > 0.6 ? 'High' : mlData.probability > 0.4 ? 'Moderate' : 'Low') : 'N/A'
  const riskColor = mlData ? (mlData.probability > 0.6 ? 'var(--red)' : mlData.probability > 0.4 ? 'var(--yellow)' : 'var(--green)') : 'var(--text-secondary)'

  return (
    <div className="view-container">
      <div className="view-header">
        <h1 className="view-title">ML Predictor</h1>
        <div className="patient-pill">Patient: {patientId}</div>
      </div>

      <div className="two-col-grid">
        {/* Prediction Card */}
        {mlData && (
          <div className="card prediction-hero">
            <div className="prediction-label">Readmission Risk Prediction</div>
            <div className="prediction-model">{mlData.model_name}</div>
            <div className="risk-gauge" style={{ '--risk-pct': `${mlData.probability * 100}%`, '--risk-color': riskColor } as React.CSSProperties}>
              <div className="risk-gauge-bar">
                <div className="risk-gauge-fill" />
              </div>
              <div className="risk-pct" style={{ color: riskColor }}>{(mlData.probability * 100).toFixed(1)}%</div>
            </div>
            <div className="risk-level" style={{ color: riskColor }}>{riskLevel} Risk</div>
            <div className="prediction-verdict">
              {mlData.prediction === 1 ? '⚠ Readmission Risk Detected' : '✓ Low Readmission Risk'}
            </div>
          </div>
        )}

        {/* Model Metrics */}
        {mlData && Object.keys(mlData.model_metrics).length > 0 && (
          <div className="card">
            <div className="card-header">Model Performance (All Candidates)</div>
            <div className="metrics-table">
              {Object.entries(mlData.model_metrics).map(([name, metrics]) => (
                <div key={name} className={`model-row ${name === mlData.model_name ? 'model-best' : ''}`}>
                  <div className="model-name">
                    {name === mlData.model_name && <span className="best-tag">BEST</span>}
                    {name}
                  </div>
                  <div className="model-metrics-row">
                    <span className="metric-chip">ROC-AUC: {metrics.roc_auc}</span>
                    <span className="metric-chip">PR-AUC: {metrics.pr_auc}</span>
                    <span className="metric-chip">F1: {metrics.f1}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* SHAP Chart */}
      {shapChartData.length > 0 && (
        <div className="card chart-card">
          <div className="card-header">SHAP Feature Contributions</div>
          <div className="chart-note">Positive values increase readmission risk; negative values decrease it</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={shapChartData} layout="vertical" margin={{ top: 5, right: 40, left: 120, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" horizontal={false} />
              <XAxis type="number" stroke="#8A9BBE" tick={{ fill: '#8A9BBE', fontSize: 11 }} />
              <YAxis type="category" dataKey="feature" stroke="#8A9BBE" tick={{ fill: '#8A9BBE', fontSize: 11 }} width={120} />
              <Tooltip
                contentStyle={{ background: '#1A2540', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                labelStyle={{ color: '#F0F4FF' }}
                formatter={(val: number) => [val.toFixed(4), 'SHAP']}
              />
              <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>
                {shapChartData.map((entry, index) => (
                  <Cell key={index} fill={entry.contribution > 0 ? 'var(--red)' : 'var(--green)'} fillOpacity={0.85} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Endpoint Evaluation */}
      {epData && (
        <div className="card">
          <div className="card-header">Trial Endpoint Evaluation</div>
          <div className="endpoints-eval">
            <div className="ep-row ep-primary">
              <div className="ep-type-label">PRIMARY</div>
              <div className="ep-name">{epData.primary.endpoint_name}</div>
              <div className="ep-change">
                {epData.primary.observed_change != null
                  ? `${epData.primary.observed_change > 0 ? '+' : ''}${epData.primary.observed_change.toFixed(2)} ${epData.primary.unit}`
                  : 'No data'}
              </div>
              <div className="ep-result" style={{ color: resultColor(epData.primary.result) }}>
                {epData.primary.result}
              </div>
            </div>
            {epData.secondary.map((ep, i) => (
              <div key={i} className="ep-row">
                <div className="ep-type-label secondary">SECONDARY</div>
                <div className="ep-name">{ep.endpoint_name}</div>
                <div className="ep-change">
                  {ep.observed_change != null
                    ? `${ep.observed_change > 0 ? '+' : ''}${ep.observed_change.toFixed(2)} ${ep.unit}`
                    : 'No data'}
                </div>
                <div className="ep-result" style={{ color: resultColor(ep.result) }}>{ep.result}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
