"""
eval_matching.py — Patient-Trial Matching Evaluation Suite
==========================================================
Evaluates the quality of the patient-trial matching pipeline using:
  - Precision@k (top-k candidates)
  - NDCG@k (ranking quality)
  - Eligibility status accuracy
  - Match score distribution analysis
  - Completeness and coverage metrics

Usage:
    python evals/eval_matching.py
    python evals/eval_matching.py --top-k 10 --output results.json
"""

import sys
import os
import json
import argparse
import math
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone

sys.path.insert(0, 'F:/PEC_Hack')

import pandas as pd

from backend.app.services.patient_matching_service import PatientMatchingService
from backend.app.services.eligibility_service import EligibilityRuleEngine


_TRIAL_DATA = {
    "trial_id": "NCT05502562",
    "condition": "Type 2 Diabetes",
    "intervention": {"name": "Oral semaglutide", "type": "drug"},
    "eligibility": {
        "inclusion": [
            "Age >= 18 years",
            "HbA1c between 7.0% and 10.5%",
            "Fasting plasma glucose < 270 mg/dL",
            "eGFR >= 60",
            "Patient must provide written informed consent",
        ],
        "exclusion": [
            "Pregnancy or breastfeeding",
            "History of pancreatitis",
            "Severe cardiovascular disease",
        ],
    },
}


def dcg_at_k(ranked_list: List[Dict], k: int, relevance_fn) -> float:
    dcg = 0.0
    for i, item in enumerate(ranked_list[:k]):
        rel = relevance_fn(item)
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(ranked_list: List[Dict], k: int, relevance_fn, ideal_rels: List[float]) -> float:
    actual_dcg = dcg_at_k(ranked_list, k, relevance_fn)
    ideal_sorted = sorted(ideal_rels, reverse=True)
    ideal_dcg = sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(ideal_sorted[:k])
    )
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def eligibility_relevance(candidate: Dict) -> float:
    status = candidate.get("eligibility_status", "NOT_ELIGIBLE")
    return {"POTENTIALLY_ELIGIBLE": 3.0, "POTENTIALLY_ELIGIBLE_WITH_REVIEW": 1.0, "NOT_ELIGIBLE": 0.0}.get(status, 0.0)


def precision_at_k(ranked_list: List[Dict], k: int) -> float:
    top_k = ranked_list[:k]
    relevant = sum(1 for c in top_k if c.get("eligibility_status") != "NOT_ELIGIBLE")
    return relevant / min(k, len(top_k)) if top_k else 0.0


def load_patients(csv_path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    result = []
    for _, row in df.iterrows():
        p = row.to_dict()
        result.append({k: (None if pd.isna(v) else v) for k, v in p.items()})
    return result


def evaluate_matching(
    csv_path: str = "F:/PEC_Hack/synthetic_type2_diabetes_trial_volunteers_200.csv",
    top_k: int = 20,
    output_path: str = None,
) -> Dict[str, Any]:
    print(f"\n{'='*60}")
    print("AI Clinical Research Assistant — Matching Evaluation Suite")
    print(f"{'='*60}")
    print(f"Trial: NCT05502562")
    print(f"Dataset: {csv_path}")
    print(f"Top-K: {top_k}")
    print(f"Evaluation started: {datetime.now(timezone.utc).isoformat()}")
    print()

    patients = load_patients(csv_path)
    print(f"Loaded {len(patients)} volunteers from dataset.\n")

    matching_service = PatientMatchingService(_TRIAL_DATA)
    ranked = matching_service.match_and_rank(patients)

    status_counts: Dict[str, int] = {}
    for c in ranked:
        s = c.get("eligibility_status", "UNKNOWN")
        status_counts[s] = status_counts.get(s, 0) + 1

    scores = [c["match_score"] for c in ranked]
    score_min = min(scores) if scores else 0
    score_max = max(scores) if scores else 0
    score_mean = sum(scores) / len(scores) if scores else 0
    score_median = sorted(scores)[len(scores)//2] if scores else 0

    ideal_rels = [eligibility_relevance(c) for c in ranked]

    p_at_k = precision_at_k(ranked, top_k)
    ndcg = ndcg_at_k(ranked, top_k, eligibility_relevance, ideal_rels)

    best = ranked[0] if ranked else {}
    best_id = best.get("patient_id", "N/A")
    best_score = best.get("match_score", 0)
    best_status = best.get("eligibility_status", "N/A")

    top_k_list = []
    for i, c in enumerate(ranked[:top_k]):
        top_k_list.append({
            "rank": i + 1,
            "patient_id": c.get("patient_id"),
            "match_score": c.get("match_score"),
            "eligibility_status": c.get("eligibility_status"),
            "passed": c.get("passed"),
            "failed": c.get("failed"),
            "unknown": c.get("unknown"),
        })

    completeness_scores = []
    key_fields = ["Age", "HbA1c_percent", "Fasting_Glucose_mg_dL", "eGFR_mL_min_1_73m2", "Weight_kg", "BMI"]
    for p in patients:
        filled = sum(1 for f in key_fields if p.get(f) is not None and str(p.get(f)).strip().lower() not in ["", "unknown", "nan"])
        completeness_scores.append(filled / len(key_fields))
    mean_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0

    results = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "trial_id": "NCT05502562",
        "total_volunteers": len(patients),
        "eligibility_distribution": status_counts,
        "ranking_metrics": {
            f"precision_at_{top_k}": round(p_at_k, 4),
            f"ndcg_at_{top_k}": round(ndcg, 4),
        },
        "score_distribution": {
            "min": round(score_min, 2),
            "max": round(score_max, 2),
            "mean": round(score_mean, 2),
            "median": round(score_median, 2),
        },
        "best_match": {
            "patient_id": best_id,
            "match_score": best_score,
            "eligibility_status": best_status,
        },
        "data_quality": {
            "mean_completeness_score": round(mean_completeness, 4),
            "fields_assessed": key_fields,
        },
        f"top_{top_k}_ranked": top_k_list,
    }

    print("=== ELIGIBILITY DISTRIBUTION ===")
    for status, count in status_counts.items():
        pct = count / len(patients) * 100
        print(f"  {status}: {count} ({pct:.1f}%)")

    print(f"\n=== RANKING QUALITY (Top-{top_k}) ===")
    print(f"  Precision@{top_k}: {p_at_k:.4f}")
    print(f"  NDCG@{top_k}:      {ndcg:.4f}")

    print(f"\n=== MATCH SCORE DISTRIBUTION ===")
    print(f"  Min: {score_min:.2f}  Max: {score_max:.2f}  Mean: {score_mean:.2f}  Median: {score_median:.2f}")

    print(f"\n=== BEST MATCHED PATIENT ===")
    print(f"  Patient ID   : {best_id}")
    print(f"  Match Score  : {best_score}")
    print(f"  Status       : {best_status}")

    print(f"\n=== DATA QUALITY ===")
    print(f"  Mean completeness: {mean_completeness:.2%}")

    print(f"\n=== TOP-{top_k} RANKED CANDIDATES ===")
    for item in top_k_list[:10]:
        print(f"  [{item['rank']:2d}] {item['patient_id']} | Score: {item['match_score']:5.1f} | {item['eligibility_status']}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patient-trial matching evaluation")
    parser.add_argument("--top-k", type=int, default=20, help="Evaluate top-K candidates")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--csv", type=str, default="F:/PEC_Hack/synthetic_type2_diabetes_trial_volunteers_200.csv")
    args = parser.parse_args()

    evaluate_matching(csv_path=args.csv, top_k=args.top_k, output_path=args.output)
