import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that improve predictive signal.
    """
    # 1. ICD-9 primary diagnosis bucketed into major disease categories
    def diag_category(code):
        try:
            c = str(code).strip()
            # V/E codes → external / supplementary
            if c.startswith('V') or c.startswith('E'):
                return 'other'
            val = float(c.split('.')[0])
            if 390 <= val <= 459 or val == 785:
                return 'circulatory'
            elif 460 <= val <= 519 or val == 786:
                return 'respiratory'
            elif 520 <= val <= 579 or val == 787:
                return 'digestive'
            elif 250 <= val <= 250.99:
                return 'diabetes'
            elif 800 <= val <= 999:
                return 'injury'
            elif 710 <= val <= 739:
                return 'musculoskeletal'
            elif 580 <= val <= 629 or val == 788:
                return 'genitourinary'
            elif 140 <= val <= 239:
                return 'neoplasms'
            else:
                return 'other'
        except Exception:
            return 'other'

    df['diag_1_cat'] = df['diag_1'].apply(diag_category)
    df['diag_2_cat'] = df['diag_2'].apply(diag_category)
    df['diag_3_cat'] = df['diag_3'].apply(diag_category)
    df['primary_diab'] = (df['diag_1_cat'] == 'diabetes').astype(int)

    # 2. High-utilization composite (strong readmission signal)
    df['total_prior_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
    df['had_prior_admission'] = (df['number_inpatient'] > 0).astype(int)

    # 3. Admission / discharge metadata
    df['admission_type_id'] = pd.to_numeric(df['admission_type_id'], errors='coerce').fillna(6).astype(int)
    df['discharge_disposition_id'] = pd.to_numeric(df['discharge_disposition_id'], errors='coerce').fillna(1).astype(int)
    df['admission_source_id'] = pd.to_numeric(df['admission_source_id'], errors='coerce').fillna(7).astype(int)

    # 4. Aggregate medication change count (number of drugs changed from baseline)
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'insulin',
    ]
    for col in med_cols:
        if col in df.columns:
            df[col] = df[col].replace('?', 'No')
    df['num_med_changes'] = df[[c for c in med_cols if c in df.columns]].apply(
        lambda row: sum(v in ('Up', 'Down') for v in row), axis=1
    )
    df['on_insulin'] = (df['insulin'].isin(['Steady', 'Up', 'Down'])).astype(int)

    return df


def prepare_data(csv_path: str, sample_size: int = None):
    """
    Load diabetic_data.csv, engineer features, define binary readmission target,
    handle missing values, and return train/test splits with a preprocessing pipeline.
    """
    df = pd.read_csv(csv_path)

    # Optional down-sample for fast test runs (None = use full dataset)
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    # Deduplicate: keep one encounter per patient (most recent)
    # Prevents data leakage from repeated admissions
    if 'patient_nbr' in df.columns:
        df = df.sort_values('encounter_id', ascending=False)
        df = df.drop_duplicates(subset='patient_nbr', keep='first')

    # Feature engineering
    df = _engineer_features(df)

    # Target: early readmission (<30 days) — clinically most actionable
    df['target'] = (df['readmitted'] == '<30').astype(int)

    # ── Feature lists ──────────────────────────────────────────────────────────
    numeric_features = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient', 'number_diagnoses',
        'total_prior_visits', 'had_prior_admission',
        'num_med_changes', 'on_insulin', 'primary_diab',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
    ]
    categorical_features = [
        'race', 'gender', 'age',
        'A1Cresult', 'max_glu_serum',
        'metformin', 'insulin', 'change', 'diabetesMed',
        'diag_1_cat', 'diag_2_cat', 'diag_3_cat',
    ]

    # Replace '?' with 'Unknown' for categorical
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].replace('?', 'Unknown')

    # Keep only existing columns
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    X = df[numeric_features + categorical_features].copy()
    y = df['target']

    # Stratified patient-level split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Preprocessing pipeline ─────────────────────────────────────────────────
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor, numeric_features, categorical_features
