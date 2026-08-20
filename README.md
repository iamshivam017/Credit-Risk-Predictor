# 📊 Credit Risk Predictor

Predicts a bank's **loan approval priority tier (P1–P4)** by merging internal
trade-line records with external CIBIL bureau data — and, along the way,
catches and corrects a data-leakage feature that would otherwise inflate
accuracy to a meaningless 99%.

---

## 📌 Problem Statement

Banks assign loan applicants a priority tier before approving credit:

| Tier | Meaning |
|---|---|
| **P1** | Lowest risk — auto-approve |
| **P2** | Low–moderate risk |
| **P3** | Moderate–high risk |
| **P4** | Highest risk — likely reject |

This project predicts that tier from an applicant's bureau/trade-line
behavior — payment history, enquiry patterns, account age — using two data
sources joined on a shared applicant ID.

## 🕵️ The Leakage Finding (why this project is more than "just a classifier")

One feature in the external dataset, `Credit_Score`, turned out to almost
perfectly determine the target — including it pushes accuracy to **99.2%**,
but that number is hollow: the project's own production/holdout file
(`unseen_dataset.xlsx`) doesn't include `Credit_Score` at all, so a model
that depends on it **can't actually be deployed**.

So this project trains and ships **two** models, both clearly labeled:

| | Model A (Full) | Model B (Deployment) |
|---|---|---|
| Features | 51 (incl. `Credit_Score`) | 50 (matches production schema) |
| Accuracy | ~99% | ~78% |
| Deployable? | ❌ No | ✅ Yes — **used for every prediction in this dashboard** |

The full story — with the exact numbers, boxplots, and reasoning — is on the
dashboard's **🕵️ Leakage Investigation** page.

---

## 📊 Dataset — Real Data, Not Synthetic

| File | Rows | Cols | What it is |
|---|---|---|---|
| `internal_bank_dataset.xlsx` | 51,336 | 26 | This bank's own trade-line/account history |
| `external_cibil_dataset.xlsx` | 51,336 | 62 | CIBIL bureau data + target `Approved_Flag` |
| `unseen_dataset.xlsx` | 100 | 42 | Production-shaped holdout — no target, no `Credit_Score` |
| `features_target_description.xlsx` | — | — | Data dictionary |

Both source files use `-99999` as a missing-value sentinel instead of `NaN` —
a common bureau-data convention that silently defeats naive `.isnull()`
checks. `src/data_cleaning.py` handles this explicitly. After cleaning and
merging on `PROSPECTID`: **42,064 clean records**.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Data handling | Pandas, NumPy, OpenPyXL |
| ML models | Scikit-learn (Logistic Regression, Random Forest), XGBoost |
| Explainability | SHAP |
| Dashboard | Streamlit |
| Visualization | Matplotlib, Seaborn |
| Testing | pytest |

---

## 📁 Folder Structure

```
CreditRiskPredictor/
├── app.py                       ← Streamlit dashboard entrypoint
├── train.py                     ← End-to-end training script (run this first)
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml              ← Dashboard theme
├── src/
│   ├── utils.py                 ← Portable paths + shared constants (single source of truth)
│   ├── data_cleaning.py         ← -99999 sentinel handling, merge logic
│   ├── feature_engineering.py   ← Encoding, feature-set definitions, schema alignment
│   ├── train_model.py           ← Model training/evaluation (LR, RF, XGBoost)
│   ├── explain.py               ← SHAP wrapper
│   ├── dashboard_utils.py       ← Cached loaders used by the dashboard
│   └── pages/
│       ├── overview.py
│       ├── predict.py           ← Single-applicant manual-entry form
│       ├── batch_scoring.py     ← CSV/Excel upload + scoring
│       ├── performance.py       ← Model comparison, confusion matrix
│       ├── explainability.py    ← SHAP plots
│       └── leakage.py           ← The Credit_Score investigation, interactively
├── data/
│   ├── raw/                     ← Source Excel files (included)
│   └── processed/               ← Auto-generated cleaning cache (gitignored)
├── models/                      ← Auto-generated: trained .pkl files + metadata.json
├── plots/                       ← Auto-generated: report-ready PNG charts
├── tests/                       ← pytest suite (33 tests)
└── docs/                        ← IEEE report, viva prep guides, notebook
    └── notebook.ipynb           ← Original analysis notebook (report source)
```

---

## ⚙️ Installation

```bash
# 1. Extract this project and enter the folder
cd CreditRiskPredictor

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

```bash
# 4. Train the models (reads data/raw/*.xlsx, saves models/*.pkl + plots/*.png)
python train.py

# 5. Launch the dashboard
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`). That's it —
no API keys, no external services, no environment variables required. Every
data file needed is already included in `data/raw/`.

**First run only:** `train.py` takes 1–3 minutes (mostly XGBoost training and
SHAP computation for the plots). Subsequent `streamlit run app.py` launches
are instant since the dashboard loads the already-trained `.pkl` files.

---

## 🖥️ Using the Dashboard

| Page | What it does |
|---|---|
| 📊 **Overview** | Headline metrics, class distribution, the two-model story |
| 🔮 **Predict** | Manual-entry form for one applicant → predicted tier + confidence |
| 📁 **Batch Scoring** | Upload a CSV/Excel (or use the built-in 100-applicant sample) → scored results, downloadable |
| 📈 **Model Performance** | All 6 model×feature-set combinations, per-class metrics, confusion matrix |
| 🔍 **Explainability (SHAP)** | Global feature importance for the deployable model |
| 🕵️ **Leakage Investigation** | The `Credit_Score` discovery, step by step, with live numbers from your trained model |

---

## 🤖 How Training Works

`train.py` reproduces the exact pipeline documented in the project report:

1. Load `internal_bank_dataset.xlsx` + `external_cibil_dataset.xlsx`
2. Clean the `-99999` sentinel (drop ~40 internal rows; drop 8 external
   columns that are >70% missing, then drop remaining sentinel rows)
3. Inner join on `PROSPECTID` → 42,064 rows × 79 columns
4. Encode features (`EDUCATION` ordinally, four categoricals one-hot)
5. Define two feature sets — Full (51 features) vs. Deployment (50 features,
   matches `unseen_dataset.xlsx`'s schema)
6. Train Logistic Regression, Random Forest, and XGBoost on **both** feature
   sets (80/20 stratified split, `random_state=42`)
7. Save the best model (XGBoost, in both settings) + a `metadata.json`
   capturing every metric the dashboard displays
8. Generate report-ready plots into `plots/`

```bash
python train.py                  # full run
python train.py --skip-plots     # skip plot generation (faster)
python train.py --no-cache       # force re-reading raw Excel files
```

> **Note on exact numbers:** XGBoost's parallel tree-building isn't always
> bit-for-bit identical across machines/library versions even with a fixed
> `random_state`. Re-running `train.py` typically reproduces accuracy within
> ~0.2–0.3 percentage points of the report's published figures (e.g. 77.7%
> vs. 77.9%) — this is expected floating-point/threading variance, not a bug.

---

## 📓 Analysis Notebook

`docs/notebook.ipynb` is the original exploratory analysis and report
source — full narrative, EDA, and the step-by-step leakage discovery in one
place. It's kept for viva/report purposes; the dashboard (`app.py`) is the
interactive way to actually use the trained model.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

33 tests across three files:
- `test_data_cleaning.py` — sentinel handling, merge correctness, dataset shape
- `test_feature_engineering.py` — encoding correctness, feature-set definitions, schema alignment (including handling of unseen categories)
- `test_model.py` — trained model sanity checks, prediction correctness, batch scoring on the real unseen dataset

`test_model.py` automatically skips if `train.py` hasn't been run yet
(no trained model to test against) — run `python train.py` first, then
the full suite passes.

---

## 🔧 Troubleshooting

**`ModuleNotFoundError` when running anything**
You're likely not inside your virtual environment, or haven't installed
requirements yet:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Dashboard shows "No trained model found"**
Run `python train.py` first — the dashboard only *reads* trained models,
it doesn't train them itself.

**Warnings about scikit-learn/XGBoost version mismatch when loading models**
The shipped models were trained and pickled with the exact versions pinned
in `requirements.txt`. Either reinstall via `pip install -r requirements.txt`,
or regenerate them against whatever you have installed: `python train.py`.

**Blank page / nothing happens in the browser**
Streamlit runs a local server — check the terminal for a `Local URL:
http://localhost:8501` line and open that manually. If the port's already
in use: `streamlit run app.py --server.port 8502`.

**Predict page gives a different tier than expected for an applicant**
Fields not shown on the form default to 0 (median-ish for a sparse bureau
record) — check the "Advanced fields" expander if you need finer control,
or use Batch Scoring with a full-schema CSV for precise control over every
one of the 42 fields.

---

## ⚠️ Limitations

- No k-fold cross-validation — reported accuracy is a single 80/20 split's
  point estimate, with no confidence interval
- No feature scaling applied before Logistic Regression (tree models are
  scale-invariant so this doesn't affect XGBoost/Random Forest, but likely
  contributes to Logistic Regression's comparatively weak ~63% accuracy)
- Class P3 is the hardest to predict (F1 ≈ 0.34) — its `Credit_Score` range
  genuinely overlaps P1, P2, and P4, a property of the label itself, not
  purely a modelling shortfall
- The `-99999` cleaning thresholds (drop column if >10,000 sentinel rows)
  are a pragmatic heuristic, not derived from a formula

## 🔮 Future Improvements

- SMOTE-based class balancing for minority classes (P1, P3, P4)
- Automated hyperparameter search (Optuna) instead of manual grid evaluation
- Target encoding for `last_prod_enq2`/`first_prod_enq2` as an alternative
  to one-hot encoding
- k-fold cross-validation for a proper confidence interval on the headline metric

---

## ✅ Project Checklist

- [x] Real bank + bureau data, cleaned and merged
- [x] Data leakage identified, quantified, and corrected for
- [x] Two models trained, compared, and clearly labeled
- [x] SHAP explainability for the deployable model
- [x] Interactive dashboard: single prediction, batch scoring, performance, explainability
- [x] 33 automated tests, all passing
- [x] Fully reproducible from raw data via one command (`python train.py`)
