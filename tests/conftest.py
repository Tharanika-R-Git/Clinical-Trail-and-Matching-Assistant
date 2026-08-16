import sys, os
sys.path.insert(0, 'F:/PEC_Hack')
import pytest

@pytest.fixture
def trial_data():
    return {
        'trial_id': 'NCT05502562',
        'condition': 'Type 2 Diabetes',
        'intervention': {'name': 'Oral semaglutide', 'type': 'drug'},
        'eligibility': {
            'inclusion': [
                'Age >= 18 years',
                'HbA1c between 7.0% and 10.5%',
                'Fasting plasma glucose < 270 mg/dL',
                'eGFR >= 60',
                'Patient must provide written informed consent'
            ],
            'exclusion': [
                'Pregnancy or breastfeeding',
                'History of pancreatitis',
                'Severe cardiovascular disease'
            ]
        }
    }

@pytest.fixture
def sample_patient():
    return {
        'Patient_ID': 'DM-TEST-001',
        'Age': 52,
        'Gender': 'Female',
        'Location': 'Mumbai',
        'HbA1c_percent': 8.4,
        'Fasting_Glucose_mg_dL': 172,
        'Weight_kg': 82,
        'BMI': 30.1,
        'eGFR_mL_min_1_73m2': 88,
        'Creatinine_mg_dl': 1.0,
        'ALT_U_L': 32,
        'Consent_for_Trial': 'Yes',
        'Pregnancy': 'No',
        'Other_Medical_Conditions': 'Hypertension',
        'Current_Medication': 'Metformin',
        'Diabetes_Type': 'Type 2 Diabetes',
        'Symptoms': 'fatigue increased thirst'
    }

@pytest.fixture
def incomplete_patient(sample_patient):
    p = sample_patient.copy()
    p['HbA1c_percent'] = None
    p['eGFR_mL_min_1_73m2'] = None
    return p

@pytest.fixture
def ineligible_patient(sample_patient):
    p = sample_patient.copy()
    p['Pregnancy'] = 'Yes'
    p['eGFR_mL_min_1_73m2'] = 30
    return p

@pytest.fixture
def sample_docs():
    return [
        'Inclusion criteria: HbA1c between 7.0% and 10.5% at screening for NCT05502562',
        'Exclusion criteria: Pregnancy or breastfeeding patients are excluded from oral semaglutide trial',
        'eGFR >= 60 mL/min/1.73m2 is required for adequate kidney function assessment',
        'Primary endpoint: Change in HbA1c from baseline to week 40 of treatment',
        'Oral semaglutide is a GLP-1 receptor agonist indicated for Type 2 Diabetes Mellitus treatment in India'
    ]
