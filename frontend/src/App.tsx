import { useState } from 'react'
import TrialView from './components/TrialView'
import MatchingRanker from './components/MatchingRanker'
import ClinicalTimeline from './components/ClinicalTimeline'
import MLPredictor from './components/MLPredictor'
import ResearchSynthesis from './components/ResearchSynthesis'

type Tab = 'trial' | 'matching' | 'timeline' | 'ml' | 'research'

interface SelectedPatient {
  patient_id: string
  match_score: number
  eligibility_status: string
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('trial')
  const [selectedPatient, setSelectedPatient] = useState<SelectedPatient | null>(null)

  const tabs: { id: Tab; label: string; icon: string }[] = [
    { id: 'trial', label: 'Trial Profile', icon: '🔬' },
    { id: 'matching', label: 'Patient Matching', icon: '🎯' },
    { id: 'timeline', label: 'Clinical Timeline', icon: '📈' },
    { id: 'ml', label: 'ML Predictor', icon: '🤖' },
    { id: 'research', label: 'Research Report', icon: '📋' },
  ]

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <span className="logo-icon">⚕</span>
            <div>
              <div className="logo-title">ClinicalAI</div>
              <div className="logo-sub">Research Assistant</div>
            </div>
          </div>
        </div>

        <nav className="sidebar-nav">
          {tabs.map(tab => (
            <button
              key={tab.id}
              id={`nav-${tab.id}`}
              className={`nav-item ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span className="nav-icon">{tab.icon}</span>
              <span className="nav-label">{tab.label}</span>
            </button>
          ))}
        </nav>

        {selectedPatient && (
          <div className="sidebar-patient-card">
            <div className="patient-card-label">Selected Patient</div>
            <div className="patient-card-id">{selectedPatient.patient_id}</div>
            <div className="patient-card-score">
              Score: <strong>{selectedPatient.match_score.toFixed(1)}</strong>
            </div>
            <div className={`badge ${selectedPatient.eligibility_status === 'POTENTIALLY_ELIGIBLE' ? 'badge-green' : 'badge-yellow'}`}>
              {selectedPatient.eligibility_status === 'POTENTIALLY_ELIGIBLE' ? 'Eligible' : 'Review'}
            </div>
          </div>
        )}

        <div className="sidebar-footer">
          <div className="trial-badge">NCT05502562</div>
          <div className="trial-badge-sub">PIONEER PLUS</div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="top-header">
          <div className="header-breadcrumb">
            <span className="breadcrumb-trial">NCT05502562</span>
            <span className="breadcrumb-sep">›</span>
            <span className="breadcrumb-tab">{tabs.find(t => t.id === activeTab)?.label}</span>
          </div>
          <div className="header-meta">
            <span className="header-tag">Phase 3b</span>
            <span className="header-tag tag-cyan">Type 2 Diabetes</span>
            <span className="header-tag tag-gold">Oral Semaglutide</span>
          </div>
        </header>

        <div className="page-content">
          {activeTab === 'trial' && <TrialView />}
          {activeTab === 'matching' && (
            <MatchingRanker
              onPatientSelect={(p) => {
                setSelectedPatient(p)
                setActiveTab('timeline')
              }}
            />
          )}
          {activeTab === 'timeline' && (
            <ClinicalTimeline patientId={selectedPatient?.patient_id || null} />
          )}
          {activeTab === 'ml' && (
            <MLPredictor patientId={selectedPatient?.patient_id || null} />
          )}
          {activeTab === 'research' && (
            <ResearchSynthesis
              patientId={selectedPatient?.patient_id || null}
              matchScore={selectedPatient?.match_score || null}
              eligibilityStatus={selectedPatient?.eligibility_status || null}
            />
          )}
        </div>
      </main>
    </div>
  )
}
