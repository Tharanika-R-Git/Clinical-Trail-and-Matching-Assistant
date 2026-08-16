"""
test_endpoint_service.py — Endpoint Service Deterministic Tests
==============================================================
Validates that endpoint evaluation is purely arithmetic and
produces correct ACHIEVED / NOT_ACHIEVED / UNKNOWN results.
"""

import sys
sys.path.insert(0, 'F:/PEC_Hack')

import pytest
from backend.app.services.endpoint_service import EndpointService


@pytest.fixture
def service():
    return EndpointService()


class TestPrimaryEndpoint:
    def test_hba1c_reduction_achieved(self, service):
        baseline = {'HbA1c_percent': 8.4, 'Weight_kg': 82, 'Fasting_Glucose_mg_dL': 172}
        followup = {'HbA1c_percent': 7.1, 'Weight_kg': 77, 'Fasting_Glucose_mg_dL': 135}
        result = service.evaluate_endpoint(baseline, followup)
        primary = result['primary']
        assert primary['result'] == 'ACHIEVED'
        assert primary['observed_change'] < 0  # negative reduction in HbA1c

    def test_hba1c_no_reduction(self, service):
        baseline = {'HbA1c_percent': 8.4, 'Weight_kg': 82, 'Fasting_Glucose_mg_dL': 172}
        followup = {'HbA1c_percent': 8.5, 'Weight_kg': 83, 'Fasting_Glucose_mg_dL': 180}
        result = service.evaluate_endpoint(baseline, followup)
        assert result['primary']['result'] == 'NOT_ACHIEVED'

    def test_small_reduction_not_achieved(self, service):
        """Less than 1.0% reduction should not achieve the primary endpoint."""
        baseline = {'HbA1c_percent': 8.0, 'Weight_kg': 82, 'Fasting_Glucose_mg_dL': 172}
        followup = {'HbA1c_percent': 7.5, 'Weight_kg': 78, 'Fasting_Glucose_mg_dL': 160}
        result = service.evaluate_endpoint(baseline, followup)
        assert result['primary']['result'] == 'NOT_ACHIEVED'

    def test_exact_threshold_achieved(self, service):
        """Exactly -1.0% change should achieve the primary endpoint."""
        baseline = {'HbA1c_percent': 8.0, 'Weight_kg': 82, 'Fasting_Glucose_mg_dL': 172}
        followup = {'HbA1c_percent': 7.0, 'Weight_kg': 78, 'Fasting_Glucose_mg_dL': 160}
        result = service.evaluate_endpoint(baseline, followup)
        assert result['primary']['result'] == 'ACHIEVED'

    def test_missing_followup_returns_unknown(self, service):
        baseline = {'HbA1c_percent': 8.4, 'Weight_kg': 82, 'Fasting_Glucose_mg_dL': 172}
        followup = {}
        result = service.evaluate_endpoint(baseline, followup)
        assert result['primary']['result'] == 'UNKNOWN'

    def test_missing_baseline_returns_unknown(self, service):
        baseline = {}
        followup = {'HbA1c_percent': 7.0}
        result = service.evaluate_endpoint(baseline, followup)
        assert result['primary']['result'] == 'UNKNOWN'


class TestSecondaryEndpoints:
    def test_secondary_endpoints_count(self, service):
        baseline = {'HbA1c_percent': 8.4, 'Weight_kg': 82, 'Fasting_Glucose_mg_dL': 172}
        followup = {'HbA1c_percent': 7.1, 'Weight_kg': 77, 'Fasting_Glucose_mg_dL': 135}
        result = service.evaluate_endpoint(baseline, followup)
        assert len(result['secondary']) == 3

    def test_weight_loss_achieved(self, service):
        baseline = {'HbA1c_percent': 8.4, 'Weight_kg': 90, 'Fasting_Glucose_mg_dL': 172}
        followup = {'HbA1c_percent': 7.0, 'Weight_kg': 85, 'Fasting_Glucose_mg_dL': 140}
        result = service.evaluate_endpoint(baseline, followup)
        weight_ep = next(ep for ep in result['secondary'] if 'weight' in ep['endpoint_name'].lower())
        assert weight_ep['result'] == 'ACHIEVED'
        assert weight_ep['observed_change'] == pytest.approx(-5.0)

    def test_fasting_glucose_reduction_achieved(self, service):
        baseline = {'HbA1c_percent': 8.4, 'Weight_kg': 82, 'Fasting_Glucose_mg_dL': 200}
        followup = {'HbA1c_percent': 7.0, 'Weight_kg': 79, 'Fasting_Glucose_mg_dL': 170}
        result = service.evaluate_endpoint(baseline, followup)
        glucose_ep = next(ep for ep in result['secondary'] if 'glucose' in ep['endpoint_name'].lower())
        assert glucose_ep['result'] == 'ACHIEVED'

    def test_hba1c_target_below_7(self, service):
        """Secondary endpoint: followup HbA1c < 7.0%."""
        baseline = {'HbA1c_percent': 8.4, 'Weight_kg': 82, 'Fasting_Glucose_mg_dL': 172}
        followup = {'HbA1c_percent': 6.8, 'Weight_kg': 78, 'Fasting_Glucose_mg_dL': 130}
        result = service.evaluate_endpoint(baseline, followup)
        target_ep = next(ep for ep in result['secondary'] if '7.0' in ep['endpoint_name'])
        assert target_ep['result'] == 'ACHIEVED'

    def test_hba1c_not_below_7(self, service):
        baseline = {'HbA1c_percent': 8.4, 'Weight_kg': 82, 'Fasting_Glucose_mg_dL': 172}
        followup = {'HbA1c_percent': 7.2, 'Weight_kg': 79, 'Fasting_Glucose_mg_dL': 155}
        result = service.evaluate_endpoint(baseline, followup)
        target_ep = next(ep for ep in result['secondary'] if '7.0' in ep['endpoint_name'])
        assert target_ep['result'] == 'NOT_ACHIEVED'


class TestClinicalChanges:
    def test_weight_change_calculation(self, service):
        baseline = {'HbA1c_percent': 8.4, 'Weight_kg': 82, 'Fasting_Glucose_mg_dL': 172}
        followup = {'HbA1c_percent': 7.1, 'Weight_kg': 77, 'Fasting_Glucose_mg_dL': 135}
        changes = service.calculate_clinical_changes(baseline, followup)
        assert abs(changes['weight_change']['change'] - (-5.0)) < 0.01

    def test_hba1c_change_calculation(self, service):
        baseline = {'HbA1c_percent': 8.4, 'Weight_kg': 82, 'Fasting_Glucose_mg_dL': 172}
        followup = {'HbA1c_percent': 7.1, 'Weight_kg': 77, 'Fasting_Glucose_mg_dL': 135}
        changes = service.calculate_clinical_changes(baseline, followup)
        assert abs(changes['hba1c_change']['change'] - (-1.3)) < 0.01

    def test_glucose_change_calculation(self, service):
        baseline = {'Fasting_Glucose_mg_dL': 200}
        followup = {'Fasting_Glucose_mg_dL': 160}
        changes = service.calculate_clinical_changes(baseline, followup)
        assert changes['glucose_change']['change'] == pytest.approx(-40.0)

    def test_missing_field_returns_none(self, service):
        baseline = {}
        followup = {}
        changes = service.calculate_clinical_changes(baseline, followup)
        assert changes['hba1c_change']['change'] is None
        assert changes['weight_change']['change'] is None

    def test_changes_deterministic(self, service):
        baseline = {'HbA1c_percent': 8.4, 'Weight_kg': 82, 'Fasting_Glucose_mg_dL': 172}
        followup = {'HbA1c_percent': 7.1, 'Weight_kg': 77, 'Fasting_Glucose_mg_dL': 135}
        r1 = service.calculate_clinical_changes(baseline, followup)
        r2 = service.calculate_clinical_changes(baseline, followup)
        assert r1['hba1c_change']['change'] == r2['hba1c_change']['change']
        assert r1['weight_change']['change'] == r2['weight_change']['change']

    def test_disclaimer_present(self, service):
        baseline = {'HbA1c_percent': 8.0}
        followup = {'HbA1c_percent': 7.0}
        result = service.evaluate_endpoint(baseline, followup)
        assert 'disclaimer' in result
