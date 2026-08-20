# Credit Risk Predictor — Project Architecture

## 1. System Overview

The **Credit Risk Predictor** is a supervised machine-learning application that predicts applicant approval-risk priority across four classes:

- **P1** — Lowest risk / auto-approve
- **P2** — Low–moderate risk
- **P3** — Moderate–high risk
- **P4** — Highest risk / likely reject

The project has two distinct execution paths:

1. **Offline training pipeline**
   - Reads internal bank data and external bureau/CIBIL-style data.
   - Cleans and merges the datasets.
   - Engineers model features.
   - Trains Logistic Regression, Random Forest, and XGBoost.
   - Compares models.
   - Saves the best deployment model.
   - Saves evaluation metadata and plots.
   - Also trains a second model containing `Credit_Score` solely to demonstrate data leakage.

2. **Online/local serving pipeline**
   - Streamlit loads the pre-trained deployment-safe XGBoost model.
   - Users can enter one applicant, upload a batch, inspect performance, view SHAP explanations, and investigate the leakage experiment.
   - New input is aligned to the exact deployment feature schema before prediction.

---

## 2. High-Level Architecture

```text
                              ┌─────────────────────────┐
                              │       RAW DATA          │
                              │                         │
                              │ internal_bank_dataset   │
                              │ external_cibil_dataset  │
                              │ unseen_dataset           │
                              │ feature descriptions     │
                              └────────────┬────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │   DATA CLEANING LAYER   │
                              │                         │
                              │ load_raw()              │
                              │ clean_internal()        │
                              │ clean_external()        │
                              │ merge_datasets()        │
                              │ load_and_clean_merged() │
                              └────────────┬────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │ FEATURE ENGINEERING     │
                              │                         │
                              │ Education → ordinal     │
                              │ Categorical → one-hot   │
                              │ Target → LabelEncoder   │
                              │ Deployment schema       │
                              │ Full schema (+CIBIL)    │
                              └────────────┬────────────┘
                                           │
                          ┌────────────────┴────────────────┐
                          │                                 │
                          ▼                                 ▼
                ┌────────────────────┐          ┌────────────────────────┐
                │ MODEL A — FULL    │          │ MODEL B — DEPLOYMENT   │
                │                   │          │                        │
                │ Features +        │          │ Features WITHOUT       │
                │ Credit_Score      │          │ Credit_Score           │
                │ (Leakage Study)   │          │ (Production-safe)      │
                └─────────┬─────────┘          └────────────┬───────────┘
                          │                                 │
                          ▼                                 ▼
                ┌────────────────────┐          ┌────────────────────────┐
                │ LR / RF / XGBoost │          │ LR / RF / XGBoost      │
                └─────────┬─────────┘          └────────────┬───────────┘
                          │                                 │
                          ▼                                 ▼
                 Evaluation Metrics                  Evaluation Metrics
                          │                                 │
                          └────────────────┬────────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │   PERSISTED ARTIFACTS   │
                              │                         │
                              │ model_deployment.pkl    │
                              │ model_full_with_        │
                              │ credit_score.pkl        │
                              │ metadata.json           │
                              │ plots/*.png             │
                              │ processed/merged_clean  │
                              └────────────┬────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │      STREAMLIT APP       │
                              │         app.py           │
                              └────────────┬────────────┘
                                           │
                    ┌──────────────────────┼───────────────────────┐
                    │                      │                       │
                    ▼                      ▼                       ▼
             Single Prediction       Batch Scoring          Performance /
             + probability          CSV/Excel upload         Leakage views
                    │
                    ▼
             ┌───────────────┐
             │ SHAP Engine   │
             │ explain.py    │
             └───────┬───────┘
                     │
                     ▼
             Feature contributions
             + confidence
             + class probabilities
```

## 3. Detailed Component Architecture

### 3.1 Data Layer

**Location:** `data/raw/`

Primary inputs:

- `internal_bank_dataset.xlsx`
- `external_cibil_dataset.xlsx`
- `unseen_dataset.xlsx`
- `features_target_description.xlsx`

The raw data is not used directly by the model. It passes through a cleaning and merge layer first.

### 3.2 Data Cleaning Layer

**File:** `src/data_cleaning.py`

Main functions:

- `load_raw()`
- `clean_internal()`
- `clean_external()`
- `merge_datasets()`
- `load_and_clean_merged()`

Processing:

```text
Internal dataset
      │
      ├── remove rows with -99999 in Age_Oldest_TL
      │
      ▼
Clean internal dataset

External bureau dataset
      │
      ├── detect columns with excessive -99999 values
      ├── drop those columns
      ├── remove remaining rows containing numeric -99999
      │
      ▼
Clean external dataset

Clean internal + clean external
      │
      └── INNER JOIN on PROSPECTID
                    │
                    ▼
             Merged dataset
```

A processed cache is stored at:

`data/processed/merged_clean.pkl`

This reduces the need to repeatedly parse the large Excel files.

### 3.3 Feature Engineering Layer

**File:** `src/feature_engineering.py`

The project uses three main transformations.

#### Education encoding

`EDUCATION` is ordinally encoded because the values have an intended order:

```text
SSC → 1
12TH → 2
UNDER GRADUATE → 3
GRADUATE / PROFESSIONAL → 4
POST-GRADUATE → 5
```

#### One-hot encoding

These categorical variables are one-hot encoded:

```text
MARITALSTATUS
GENDER
last_prod_enq2
first_prod_enq2
```

`drop_first=True` is used.

#### Target encoding

`Approved_Flag` is transformed with `LabelEncoder` into the four model classes:

```text
P1, P2, P3, P4
```

### 3.4 Two Feature Sets

The architecture intentionally maintains two feature sets.

#### Deployment Feature Set

Used for real prediction.

It contains approximately **49 model input features after categorical expansion** and **does not contain `Credit_Score`**.

Reason:

`Credit_Score` is unavailable in the project's intended unseen scoring data and is therefore treated as leakage for deployment.

#### Full Feature Set

Deployment features +:

```text
Credit_Score
```

This version is used only for the leakage investigation.

---

## 4. Training Architecture

**Entrypoint:** `train.py`

```text
Raw Excel
   │
   ▼
load_and_clean_merged()
   │
   ▼
engineer_features()
   │
   ▼
get_feature_sets()
   │
   ├───────────────┐
   │               │
   ▼               ▼
Full features   Deployment features
   │               │
   └───────┬───────┘
           ▼
   Stratified 80/20 split
           │
           ▼
 ┌──────────────────────────────┐
 │ Train 3 candidate classifiers│
 │                              │
 │ 1. Logistic Regression       │
 │ 2. Random Forest             │
 │ 3. XGBoost                   │
 └──────────────┬───────────────┘
                │
                ▼
      Accuracy + Macro F1
                │
                ▼
          Select XGBoost
                │
       ┌────────┴────────┐
       ▼                 ▼
 Full model         Deployment model
       │                 │
       ▼                 ▼
joblib .pkl         joblib .pkl
```

### Model hyperparameters

#### XGBoost

- `n_estimators=300`
- `max_depth=5`
- `learning_rate=0.08`
- `subsample=0.9`
- `colsample_bytree=0.9`
- `min_child_weight=3`
- `eval_metric=mlogloss`

#### Random Forest

- `n_estimators=200`
- `max_depth=10`
- `n_jobs=-1`

#### Logistic Regression

- `max_iter=1000`

---

## 5. Model Results Captured in the Current Project

The stored `metadata.json` reports:

| Model | Feature set | Accuracy | Macro F1 |
|---|---|---:|---:|
| Logistic Regression | Full | 0.6459 | 0.3520 |
| Random Forest | Full | 0.9851 | 0.9745 |
| XGBoost | Full | 0.9927 | 0.9863 |
| Logistic Regression | Deployment | 0.6313 | 0.3464 |
| Random Forest | Deployment | 0.7614 | 0.6185 |
| **XGBoost** | **Deployment** | **0.7768** | **0.6859** |

The important architectural conclusion is that the full model's extremely high score is not the model intended for live use. The deployment model intentionally removes `Credit_Score` and reports a more realistic performance level.

Current merged training dataset:

- **42,064 rows**
- **79 columns**
- Target classes: P1–P4

---

## 6. Leakage Investigation Architecture

The project deliberately contains a controlled experiment:

```text
                  Same cleaned training data
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      With Credit_Score            Without Credit_Score
             │                           │
             ▼                           ▼
        XGBoost 99.27%              XGBoost 77.68%
```

This demonstrates why `Credit_Score` should not be treated as an available online feature in this project's deployment scenario.

The Streamlit **Leakage Investigation** page exposes this comparison.

---

## 7. Serving / Inference Architecture

**Entrypoint:** `app.py`

At application startup:

```text
Streamlit app.py
    │
    ├── load_metadata()
    ├── load_deployment_model()
    └── load page modules
```

The deployment model is loaded from:

`models/model_deployment.pkl`

The model is cached with Streamlit's `@st.cache_resource`, so it is not repeatedly loaded on every interaction.

### Single applicant prediction

```text
User form
   │
   ▼
Raw applicant dictionary
   │
   ▼
predict_single()
   │
   ▼
align_to_deployment_schema()
   │
   ├── education mapping
   ├── one-hot encoding
   └── exact column reindexing
   │
   ▼
XGBoost.predict()
XGBoost.predict_proba()
   │
   ▼
P1/P2/P3/P4
+ confidence
+ class probabilities
```

### Batch prediction

```text
CSV / Excel upload
      │
      ▼
predict_batch()
      │
      ▼
align_to_deployment_schema()
      │
      ▼
XGBoost inference
      │
      ▼
Original rows +
predicted_class +
confidence
```

---

## 8. Explainability Architecture

**File:** `src/explain.py`

The project uses **SHAP TreeExplainer** for the tree-based model.

```text
Deployment XGBoost
       │
       ▼
SHAP TreeExplainer
       │
       ├── global feature importance
       ├── class-specific importance
       └── single applicant explanation
                │
                ▼
       Top contributing features
       + SHAP values
       + applicant values
       + predicted class
       + confidence
       + class probabilities
```

This gives the application an interpretable layer rather than returning only a class label.

---

## 9. Application Pages

The Streamlit application currently exposes:

| Page | Purpose |
|---|---|
| Overview | Dataset, model and project summary |
| Predict | Single applicant risk prediction |
| Batch Scoring | Upload and score multiple applicants |
| Model Performance | Compare model metrics and evaluation results |
| Explainability (SHAP) | Inspect feature importance and explanations |
| Leakage Investigation | Compare full vs deployment feature sets |

The page implementations are under:

`src/pages/`

---

## 10. Persisted Artifacts

```text
models/
├── model_deployment.pkl
├── model_full_with_credit_score.pkl
└── metadata.json

data/processed/
└── merged_clean.pkl

plots/
├── 01_target_distribution.png
├── 02_credit_score_leakage.png
├── 03_model_comparison.png
├── 04_confusion_matrix.png
├── 05_shap_summary.png
├── 06_feature_distributions.png
└── 07_shap_p4_class.png
```

### Artifact responsibilities

- `model_deployment.pkl` — live prediction model.
- `model_full_with_credit_score.pkl` — leakage-study model.
- `metadata.json` — feature schema, model metrics, class definitions and training metadata.
- `merged_clean.pkl` — cleaned/merged dataset cache.
- `plots/` — training and analysis visuals.

---

## 11. Testing Architecture

**Directory:** `tests/`

The tests cover:

```text
Data cleaning
   │
   ├── expected sentinel handling
   └── merged data behavior

Feature engineering
   │
   ├── encoding
   ├── feature alignment
   └── deployment schema

Model
   │
   ├── correct model features
   ├── four output classes
   ├── metadata metrics
   ├── single prediction
   └── batch prediction
```

A key architectural test is that:

```text
Deployment model
      MUST NOT contain Credit_Score
```

while the leakage-study model:

```text
Full model
      MUST contain Credit_Score
```

---

## 12. Architecture Strengths

### Strong separation of concerns

The project separates:

- data cleaning
- feature engineering
- model training
- explainability
- dashboard utilities
- UI pages
- tests

### Deployment-safe schema alignment

`align_to_deployment_schema()` reindexes incoming data to exactly match the trained model's feature columns. Missing one-hot columns are filled with zero.

### Leakage awareness

The project does not hide the very high `Credit_Score` result. Instead, it explicitly demonstrates the difference between the full and deployment-safe models.

### Reproducibility

`RANDOM_STATE = 42` and pinned dependency versions make the project easier to reproduce.

---

## 13. Important Technical Observations / Limitations

### 13.1 The project currently persists the preprocessing logic implicitly rather than as a single sklearn Pipeline

Training transformations are coded in `feature_engineering.py`, while model serialization stores the trained XGBoost estimator.

This works for the current application because the same Python transformation code runs before inference, but a more production-grade design would package preprocessing and model inference into one serialized `Pipeline` or equivalent artifact.

### 13.2 The numeric encoding rules are hard-coded

Mappings such as `EDUCATION_ORDER` live in `utils.py`.

This is consistent and safe for the current project, but a versioned preprocessing artifact would be more robust for long-term deployment.

### 13.3 Data cache documentation is inconsistent

`MERGED_CACHE_PATH` points to:

`merged_clean.pkl`

and the implementation uses `pd.read_pickle()` / `to_pickle()`.

Some docstrings refer to a parquet cache. That documentation should be corrected to say **Pickle cache**.

### 13.4 Probability confidence is model probability, not calibrated risk

The dashboard's `confidence` comes directly from:

`XGBClassifier.predict_proba()`

That is a probability estimate, not necessarily a calibrated probability of default/approval. A production credit-risk system would normally evaluate calibration and business thresholds separately.

### 13.5 Class imbalance is visible

The merged dataset is unevenly distributed, with P2 substantially larger than the other classes. Macro F1 is therefore important in addition to accuracy. The deployment XGBoost model's weaker P3 performance is a meaningful limitation.

---

## 14. Recommended Production Architecture

For a more production-ready version, evolve the current architecture to:

```text
                 ┌──────────────────┐
                 │ Data Sources     │
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │ Validation       │
                 │ + Data Quality   │
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │ Preprocessing    │
                 │ sklearn Pipeline │
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │ XGBoost Model    │
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │ Model Registry   │
                 │ + Versioning     │
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │ Prediction API   │
                 │ FastAPI          │
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐
                 │ Web Dashboard    │
                 │ Streamlit/UI     │
                 └──────────────────┘

              Supporting layers:
              ─ monitoring
              ─ drift detection
              ─ calibration
              ─ audit logs
              ─ access control
```

This is an evolution of the current project, not a description of components that already exist.

---

## 15. Final Architecture Summary

**Current project architecture:**

```text
Excel Data
   ↓
Cleaning & Merge
   ↓
Feature Engineering
   ↓
Deployment / Full Feature Split
   ↓
Train LR + RF + XGBoost
   ↓
Evaluate
   ↓
Persist XGBoost Models + Metadata
   ↓
Streamlit Dashboard
   ↓
Single Prediction / Batch Scoring
   ↓
SHAP Explainability
```

The most important architectural decision is the separation between:

**Model A — Full / leakage investigation**

and

**Model B — Deployment / no `Credit_Score`**

The **deployment-safe XGBoost model** is the actual serving model used by the dashboard.
