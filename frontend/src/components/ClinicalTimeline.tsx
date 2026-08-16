import { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

interface ChangeData {
  hba1c_change: { baseline: number | null; followup: number | null; change: number | null; unit: string }
  weight_change: { baseline: number | null; followup: number | null; change: number | null; unit: string }
  glucose_change: { baseline: number | null; followup: number | null; change: number | null; unit: string }
  bmi_change: { baseline: number | null; followup: number | null; change: number | null; unit: string }
  bp_systolic_change: { baseline: number | null; followup: number | null; change: number | null; unit: string }
  source: string
}

interface Props {
  patientId: string | null
}

type ChartPoint = { timepoint: string; HbA1c: number | null; Weight: number | null; Glucose: number | null }

export default function ClinicalTimeline({ patientId }: Props) {
  const [data, setData] = useState<ChangeData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!patientId) return
    setLoading(true)
    setError(null)
    fetch('/api/analysis/clinical-changes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ patient_id: patientId })
    })
      .then(r => r.json())
      .then(d => { setData(d.changes); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [patientId])

  if (!patientId) return (
    <div className="view-container">
      <div className="view-header"><h1 className="view-title">Clinical Timeline</h1></div>
      <div className="empty-state">
        <div className="empty-icon">📈</div>
        <div className="empty-title">No Patient Selected</div>
        <div className="empty-sub">Go to Patient Matching and select a candidate to view their clinical timeline.</div>
      </div>
    </div>
  )

  if (loading) return <div className="loading-state"><div className="spinner" />Loading clinical data...</div>
  if (error) return <div className="error-state">⚠ {error}</div>

  const formatChange = (val: number | null, unit: string) => {
    if (val == null) return 'N/A'
    return `${val > 0 ? '+' : ''}${val.toFixed(2)} ${unit}`
  }

  const changeClass = (val: number | null) => {
    if (val == null) return 'neutral'
    return val < 0 ? 'positive-change' : 'negative-change'
  }

  // Build chart data
  const chartData: ChartPoint[] = data ? [
    {
      timepoint: 'Baseline',
      HbA1c: data.hba1c_change.baseline,
      Weight: data.weight_change.baseline,
      Glucose: data.glucose_change.baseline,
    },
    {
      timepoint: 'Week 40',
      HbA1c: data.hba1c_change.followup,
      Weight: data.weight_change.followup,
      Glucose: data.glucose_change.followup,
    }
  ] : []

  return (
    <div className="view-container">
      <div className="view-header">
        <h1 className="view-title">Clinical Timeline</h1>
        <div className="patient-pill">Patient: {patientId}</div>
      </div>

      {data && (
        <>
          {/* Change Cards */}
          <div className="metrics-grid">
            {[
              { label: 'HbA1c Change', change: data.hba1c_change.change, unit: '%', baseline: data.hba1c_change.baseline, followup: data.hba1c_change.followup, icon: '🩸' },
              { label: 'Weight Change', change: data.weight_change.change, unit: 'kg', baseline: data.weight_change.baseline, followup: data.weight_change.followup, icon: '⚖️' },
              { label: 'Fasting Glucose', change: data.glucose_change.change, unit: 'mg/dL', baseline: data.glucose_change.baseline, followup: data.glucose_change.followup, icon: '🔬' },
              { label: 'BMI Change', change: data.bmi_change.change, unit: 'kg/m²', baseline: data.bmi_change.baseline, followup: data.bmi_change.followup, icon: '📏' },
              { label: 'BP Systolic', change: data.bp_systolic_change.change, unit: 'mmHg', baseline: data.bp_systolic_change.baseline, followup: data.bp_systolic_change.followup, icon: '💓' },
            ].map(metric => (
              <div key={metric.label} className="metric-card">
                <div className="metric-icon">{metric.icon}</div>
                <div className="metric-label">{metric.label}</div>
                <div className={`metric-change ${changeClass(metric.change)}`}>
                  {formatChange(metric.change, metric.unit)}
                </div>
                <div className="metric-row">
                  <span className="metric-bl">Baseline: {metric.baseline ?? 'N/A'}</span>
                  <span className="metric-arr">→</span>
                  <span className="metric-fu">W40: {metric.followup ?? 'N/A'}</span>
                </div>
              </div>
            ))}
          </div>

          {/* HbA1c + Weight chart */}
          <div className="card chart-card">
            <div className="card-header">Biomarker Trajectory (Baseline → Week 40)</div>
            <div className="chart-note">Simulated follow-up based on GLP-1 population response data</div>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                <XAxis dataKey="timepoint" stroke="#8A9BBE" tick={{ fill: '#8A9BBE', fontSize: 12 }} />
                <YAxis yAxisId="left" stroke="#00D4FF" tick={{ fill: '#00D4FF', fontSize: 11 }} label={{ value: 'HbA1c %', angle: -90, position: 'insideLeft', fill: '#00D4FF', fontSize: 11 }} />
                <YAxis yAxisId="right" orientation="right" stroke="#00E676" tick={{ fill: '#00E676', fontSize: 11 }} label={{ value: 'Weight kg', angle: 90, position: 'insideRight', fill: '#00E676', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: '#1A2540', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }}
                  labelStyle={{ color: '#F0F4FF' }}
                />
                <Legend wrapperStyle={{ color: '#8A9BBE', paddingTop: 12 }} />
                <Line yAxisId="left" type="monotone" dataKey="HbA1c" stroke="#00D4FF" strokeWidth={2.5} dot={{ fill: '#00D4FF', r: 5 }} />
                <Line yAxisId="right" type="monotone" dataKey="Weight" stroke="#00E676" strokeWidth={2.5} dot={{ fill: '#00E676', r: 5 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="disclaimer-card">
            <span className="disclaimer-icon">ℹ</span>
            {data.source}
          </div>
        </>
      )}
    </div>
  )
}
