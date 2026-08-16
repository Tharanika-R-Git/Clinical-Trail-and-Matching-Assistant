import { useState, useRef } from 'react'

interface ExtractedField {
  value: number | string
  unit: string | null
  confidence: 'HIGH' | 'MEDIUM' | 'LOW'
  raw_match: string
}

interface EligibilityCriterion {
  criterion: string
  type: string
  status: 'PASS' | 'FAIL' | 'UNKNOWN'
  patient_value?: number | string | null
}

interface EligibilityResult {
  eligibility_status: string
  passed: number
  failed: number
  unknown: number
  criteria_detail: EligibilityCriterion[]
}

interface OcrResult {
  filename: string
  raw_text: string
  extracted_fields: Record<string, ExtractedField>
  unparsed_count: number
  eligibility: EligibilityResult | null
  eligibility_note: string | null
}

const eligibilityColor = (s: string) => {
  if (s === 'POTENTIALLY_ELIGIBLE') return 'badge-green'
  if (s === 'POTENTIALLY_ELIGIBLE_WITH_REVIEW') return 'badge-yellow'
  return 'badge-red'
}
const eligibilityLabel = (s: string) => {
  if (s === 'POTENTIALLY_ELIGIBLE') return 'Eligible'
  if (s === 'POTENTIALLY_ELIGIBLE_WITH_REVIEW') return 'Review Required'
  return 'Ineligible'
}

const criterionColor = (s: string) => {
  if (s === 'PASS') return 'badge-green'
  if (s === 'FAIL') return 'badge-red'
  return 'badge-muted'
}

export default function LabReportOCR() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<OcrResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = (selected: File | null) => {
    if (!selected) return
    setFile(selected)
    setResult(null)
    setError(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(URL.createObjectURL(selected))
  }

  const extract = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const r = await fetch('/api/ocr/extract', { method: 'POST', body: formData })
      if (!r.ok) {
        const err = await r.json().catch(() => null)
        throw new Error(err?.detail || `API error ${r.status}`)
      }
      setResult((await r.json()) as OcrResult)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const fields = Object.entries(result?.extracted_fields || {})

  return (
    <div className="view-container">
      <div className="view-header">
        <h1 className="view-title">Lab Report OCR</h1>
        <div className="header-meta">
          <span className="header-tag">Image → Text</span>
          <span className="header-tag tag-cyan">Structured Labs</span>
          <span className="header-tag tag-gold">Eligibility Check</span>
        </div>
      </div>

      {error && <div className="error-banner">⚠ {error}</div>}

      {/* Upload */}
      <div className="card">
        <div className="card-header">Upload Lab Report Image</div>
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp,image/bmp,image/tiff"
          style={{ display: 'none' }}
          onChange={(e) => handleFile(e.target.files?.[0] || null)}
        />
        <div
          className="ocr-dropzone"
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault()
            handleFile(e.dataTransfer.files?.[0] || null)
          }}
        >
          {previewUrl ? (
            <div className="ocr-preview">
              <img src={previewUrl} alt="Lab report preview" />
              <div className="ocr-preview-name">{file?.name}</div>
            </div>
          ) : (
            <>
              <div className="ocr-dropzone-icon">📄</div>
              <div className="ocr-dropzone-title">Drop a lab report image here</div>
              <div className="ocr-dropzone-sub">PNG, JPEG, WEBP, BMP or TIFF — or click to browse</div>
            </>
          )}
        </div>
        {file && (
          <div className="ocr-actions">
            <button id="ocr-extract-btn" className="btn-primary" onClick={extract} disabled={loading}>
              {loading ? <><span className="spinner-sm" /> Extracting...</> : '🔍 Extract & Evaluate'}
            </button>
            <button
              className="btn-secondary"
              onClick={() => {
                setFile(null)
                setResult(null)
                if (previewUrl) URL.revokeObjectURL(previewUrl)
                setPreviewUrl(null)
              }}
            >
              Clear
            </button>
          </div>
        )}
      </div>

      {/* Raw text */}
      {result && result.raw_text && (
        <div className="card">
          <div className="card-header">Raw OCR Text</div>
          <pre className="ocr-raw-text">{result.raw_text}</pre>
        </div>
      )}

      {loading && (
        <div className="loading-state">
          <div className="spinner" />
          Running OCR and eligibility pipeline...
        </div>
      )}

      {/* Results */}
      {result && !loading && (
        <>
          <div className="two-col-grid">
            {/* Extracted fields */}
            <div className="card">
              <div className="card-header">
                Extracted Lab Values ({fields.length})
                {result.unparsed_count > 0 && (
                  <span className="ocr-unparsed"> · {result.unparsed_count} fields not detected</span>
                )}
              </div>
              {fields.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-icon">🧪</div>
                  <div className="empty-title">No values detected</div>
                  <div className="empty-sub">Try a clearer or higher-resolution image.</div>
                </div>
              ) : (
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                        <th>Unit</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {fields.map(([key, f]) => (
                        <tr key={key} className="table-row">
                          <td className="mono">{key}</td>
                          <td className="font-semibold">{f.value}</td>
                          <td>{f.unit ?? '–'}</td>
                          <td>
                            <span className={`badge ${f.confidence === 'HIGH' ? 'badge-green' : 'badge-yellow'}`}>
                              {f.confidence}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            {/* Eligibility */}
            <div className="card">
              <div className="card-header">Trial Eligibility (NCT05502562)</div>
              {result.eligibility ? (
                <>
                  <div className="ocr-eligibility-hero">
                    <span className={`badge ${eligibilityColor(result.eligibility.eligibility_status)}`}>
                      {eligibilityLabel(result.eligibility.eligibility_status)}
                    </span>
                    <div className="ocr-eligibility-counts">
                      <span className="sum-green">✓ {result.eligibility.passed} pass</span>
                      <span className="sum-red">✗ {result.eligibility.failed} fail</span>
                      <span className="sum-label">? {result.eligibility.unknown} unknown</span>
                    </div>
                  </div>
                  <div className="criteria-list">
                    {result.eligibility.criteria_detail.map((c, i) => (
                      <div key={i} className="criteria-item">
                        <span className={`badge ${criterionColor(c.status)}`}>{c.status}</span>
                        <span className="criteria-text">{c.criterion}</span>
                        <span className="criteria-value">
                          {c.patient_value != null && c.patient_value !== '' ? String(c.patient_value) : '–'}
                        </span>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <div className="empty-state">
                  <div className="empty-icon">⚠</div>
                  <div className="empty-title">Cannot evaluate eligibility</div>
                  <div className="empty-sub">{result.eligibility_note || 'No lab values were extracted.'}</div>
                </div>
              )}
            </div>
          </div>

          <div className="disclaimer-card">
            <span className="disclaimer-icon">ℹ</span>
            <span className="disclaimer-text">
              Extracted values are based on automated OCR and require human verification before any clinical use.
              {result.eligibility_note}
            </span>
          </div>
        </>
      )}

      {!result && !loading && (
        <div className="empty-state">
          <div className="empty-icon">🧪</div>
          <div className="empty-title">Upload a Lab Report</div>
          <div className="empty-sub">
            Upload a photo or scan of a lab report. OCR extracts structured lab values and checks them against trial
            eligibility criteria.
          </div>
        </div>
      )}
    </div>
  )
}
