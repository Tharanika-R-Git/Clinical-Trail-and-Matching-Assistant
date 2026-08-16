import sys
sys.path.insert(0, 'F:/PEC_Hack')
import pytest
from backend.app.services.eligibility_service import EligibilityRuleEngine


def test_age_pass(trial_data):
    engine = EligibilityRuleEngine(trial_data)
    patient = {'Patient_ID': 'T01', 'Age': 52, 'Consent_for_Trial': 'Yes', 'Pregnancy': 'No', 'Other_Medical_Conditions': 'None'}
    result = engine.evaluate_patient(patient)
    age_criterion = next(c for c in result['criteria_detail'] if c['criterion_id'] == 'INC-001')
    assert age_criterion['status'] == 'PASS'


def test_age_fail(trial_data):
    engine = EligibilityRuleEngine(trial_data)
    patient = {'Patient_ID': 'T02', 'Age': 15, 'Consent_for_Trial': 'Yes', 'Pregnancy': 'No', 'Other_Medical_Conditions': 'None'}
    result = engine.evaluate_patient(patient)
    age_criterion = next(c for c in result['criteria_detail'] if c['criterion_id'] == 'INC-001')
    assert age_criterion['status'] == 'FAIL'


def test_age_missing(trial_data):
    engine = EligibilityRuleEngine(trial_data)
    patient = {'Patient_ID': 'T03', 'Age': None, 'Consent_for_Trial': 'Yes', 'Pregnancy': 'No', 'Other_Medical_Conditions': 'None'}
    result = engine.evaluate_patient(patient)
    age_criterion = next(c for c in result['criteria_detail'] if c['criterion_id'] == 'INC-001')
    assert age_criterion['status'] == 'UNKNOWN'


def test_hba1c_in_range(trial_data):
    engine = EligibilityRuleEngine(trial_data)
    patient = {'Patient_ID': 'T04', 'Age': 52, 'HbA1c_percent': 8.4, 'Consent_for_Trial': 'Yes', 'Pregnancy': 'No', 'Other_Medical_Conditions': 'None'}
    result = engine.evaluate_patient(patient)
    hba1c_criterion = next(c for c in result['criteria_detail'] if c['criterion_id'] == 'INC-002')
    assert hba1c_criterion['status'] == 'PASS'


def test_hba1c_too_high(trial_data):
    engine = EligibilityRuleEngine(trial_data)
    patient = {'Patient_ID': 'T05', 'Age': 52, 'HbA1c_percent': 11.5, 'Consent_for_Trial': 'Yes', 'Pregnancy': 'No', 'Other_Medical_Conditions': 'None'}
    result = engine.evaluate_patient(patient)
    hba1c_criterion = next(c for c in result['criteria_detail'] if c['criterion_id'] == 'INC-002')
    assert hba1c_criterion['status'] == 'FAIL'


def test_hba1c_missing(trial_data):
    engine = EligibilityRuleEngine(trial_data)
    patient = {'Patient_ID': 'T06', 'Age': 52, 'HbA1c_percent': None, 'Consent_for_Trial': 'Yes', 'Pregnancy': 'No', 'Other_Medical_Conditions': 'None'}
    result = engine.evaluate_patient(patient)
    hba1c_criterion = next(c for c in result['criteria_detail'] if c['criterion_id'] == 'INC-002')
    assert hba1c_criterion['status'] == 'UNKNOWN'


def test_egfr_adequate(trial_data):
    engine = EligibilityRuleEngine(trial_data)
    patient = {'Patient_ID': 'T07', 'Age': 52, 'eGFR_mL_min_1_73m2': 88, 'Consent_for_Trial': 'Yes', 'Pregnancy': 'No', 'Other_Medical_Conditions': 'None'}
    result = engine.evaluate_patient(patient)
    egfr_criterion = next(c for c in result['criteria_detail'] if c['criterion_id'] == 'INC-004')
    assert egfr_criterion['status'] == 'PASS'


def test_egfr_renal_impairment(trial_data):
    engine = EligibilityRuleEngine(trial_data)
    patient = {'Patient_ID': 'T08', 'Age': 52, 'eGFR_mL_min_1_73m2': 45, 'Consent_for_Trial': 'Yes', 'Pregnancy': 'No', 'Other_Medical_Conditions': 'None'}
    result = engine.evaluate_patient(patient)
    egfr_criterion = next(c for c in result['criteria_detail'] if c['criterion_id'] == 'INC-004')
    assert egfr_criterion['status'] == 'FAIL'


def test_egfr_missing(trial_data):
    engine = EligibilityRuleEngine(trial_data)
    patient = {'Patient_ID': 'T09', 'Age': 52, 'eGFR_mL_min_1_73m2': None, 'Consent_for_Trial': 'Yes', 'Pregnancy': 'No', 'Other_Medical_Conditions': 'None'}
    result = engine.evaluate_patient(patient)
    egfr_criterion = next(c for c in result['criteria_detail'] if c['criterion_id'] == 'INC-004')
    assert egfr_criterion['status'] == 'UNKNOWN'


def test_consent_missing(trial_data):
    engine = EligibilityRuleEngine(trial_data)
    patient = {'Patient_ID': 'T10', 'Age': 52, 'Consent_for_Trial': 'No', 'Pregnancy': 'No', 'Other_Medical_Conditions': 'None'}
    result = engine.evaluate_patient(patient)
    consent_criterion = next(c for c in result['criteria_detail'] if c['criterion_id'] == 'INC-005')
    assert consent_criterion['status'] == 'FAIL'


def test_consent_provided(trial_data):
    engine = EligibilityRuleEngine(trial_data)
    patient = {'Patient_ID': 'T11', 'Age': 52, 'Consent_for_Trial': 'Yes', 'Pregnancy': 'No', 'Other_Medical_Conditions': 'None'}
    result = engine.evaluate_patient(patient)
    consent_criterion = next(c for c in result['criteria_detail'] if c['criterion_id'] == 'INC-005')
    assert consent_criterion['status'] == 'PASS'


def test_pregnancy_excluded(trial_data):
    engine = EligibilityRuleEngine(trial_data)
    patient = {'Patient_ID': 'T12', 'Age': 52, 'Consent_for_Trial': 'Yes', 'Pregnancy': 'Yes', 'Other_Medical_Conditions': 'None'}
    result = engine.evaluate_patient(patient)
    preg_criterion = next(c for c in result['criteria_detail'] if c['criterion_id'] == 'EXC-001')
    assert preg_criterion['status'] == 'FAIL'


def test_not_pregnant_pass(trial_data):
    engine = EligibilityRuleEngine(trial_data)
    patient = {'Patient_ID': 'T13', 'Age': 52, 'Consent_for_Trial': 'Yes', 'Pregnancy': 'No', 'Other_Medical_Conditions': 'None'}
    result = engine.evaluate_patient(patient)
    preg_criterion = next(c for c in result['criteria_detail'] if c['criterion_id'] == 'EXC-001')
    assert preg_criterion['status'] == 'PASS'


def test_full_eligible_patient(trial_data, sample_patient):
    engine = EligibilityRuleEngine(trial_data)
    result = engine.evaluate_patient(sample_patient)
    assert result['eligibility_status'] in ['POTENTIALLY_ELIGIBLE', 'POTENTIALLY_ELIGIBLE_WITH_REVIEW']
    assert result['failed'] == 0


def test_full_ineligible_patient(trial_data, ineligible_patient):
    engine = EligibilityRuleEngine(trial_data)
    result = engine.evaluate_patient(ineligible_patient)
    assert result['eligibility_status'] == 'NOT_ELIGIBLE'


def test_partial_unknown_patient(trial_data, incomplete_patient):
    engine = EligibilityRuleEngine(trial_data)
    result = engine.evaluate_patient(incomplete_patient)
    assert result['unknown'] > 0
    assert result['eligibility_status'] == 'POTENTIALLY_ELIGIBLE_WITH_REVIEW'
