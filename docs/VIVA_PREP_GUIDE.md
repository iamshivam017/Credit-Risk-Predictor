# Viva Preparation Guide
## Credit Risk Prediction — Bank Approval Priority Classification (P1–P4)
**Shivam Kumar | 23035010286 | IIIT Guwahati**

---

## ⚠️ Read this first — an honest scope note

Your project is a **Jupyter notebook–based data science/ML analysis project**: data cleaning →
feature engineering → multiclass classification → interpretability → scoring a holdout set.
It is **not** a full-stack application. That means several categories in a generic viva-prep
checklist genuinely **do not apply** to your project, and inventing answers for them would hurt
you more than saying "not applicable." Flagging these honestly, as instructed:

| Category often expected | Status in your project |
|---|---|
| System architecture diagram / UML / flowcharts | **Not present.** Your "architecture" is the notebook's linear pipeline (Sections 1–15). If asked for a diagram, describe the pipeline verbally (see Section 7 below). |
| Frontend / backend / APIs | **Not present.** No web app, no API, no authentication. |
| Database / schema | **Not present.** Data sources are Excel/CSV files read directly into Pandas. |
| Screenshots / UI | **Not present.** |
| Formal unit testing / test cases | **Not present.** Validation was done via train/test split metrics and manual sanity checks (e.g., confirming SHAP tensor shape, confirming feature-name alignment with `unseen_dataset.xlsx`), not a test suite. Say this plainly if asked — do not invent test cases. |
| Deployment | **Not present.** Two `.pkl` model files are saved via `joblib`, but there is no deployed service. "Deployment-safe" in your report means *feature-schema-compatible with the unseen dataset*, not "deployed to production." |

If an examiner asks about any of these, the safe answer is: *"This is an analytical ML pipeline built in a notebook, not a deployed application, so [X] wasn't part of the scope — my focus was on the modelling and leakage-detection methodology."* That is a legitimate, defensible answer — do not panic or invent one.

---

# 1. How to Explain the Whole Project in 60 Seconds (memorize this)

> "A bank wants to sort loan applicants into four risk tiers — P1 best, P4 worst — using its own internal trade-line records plus external CIBIL bureau data. I merged the two datasets, cleaned a `-99999` missing-value sentinel, and while exploring the data found that one bureau feature, `Credit_Score`, almost perfectly determines the target — and it's the one column excluded from the project's own unseen/holdout file. So instead of reporting one number, I trained two models: one with `Credit_Score` that hits 99% accuracy but can't be deployed, and one without it — the honest, deployable model — that gets 78% accuracy and 0.69 macro-F1 using XGBoost. I used SHAP to explain that model, confirming it relies on real behavioral signals like trade-line age and recent enquiries, not a shortcut. Then I scored the actual unseen applicants with it."

Practice saying this without notes. Everything else in this guide expands on pieces of it.

---

# 2. Section-by-Section Question Bank

## 2.1 Abstract / Introduction

| Question | Short Answer |
|---|---|
| What is this project about? | Predicting a bank's 4-tier loan approval priority (P1–P4) using internal + external bureau data. |
| What problem does it solve? | Automates applicant risk-tiering from raw behavioral data instead of manual review. |
| What's the main contribution? | Not just a model — the discovery and correction of a leakage feature (`Credit_Score`), and training two clearly-labeled models to show the honest tradeoff. |
| Why is this a 4-class problem, not binary (default/no-default)? | The bank's actual label `Approved_Flag` has 4 levels — it's a triage system, not a binary accept/reject. |
| What methodology did you use overall? | Data cleaning → merge → EDA → leakage investigation → feature engineering → train 3 classifiers on 2 feature sets → evaluate → SHAP → score unseen data. |
| What's the headline result? | XGBoost, deployment-safe (no `Credit_Score`): 77.9% accuracy, 0.688 macro-F1. |

**Follow-up chain:**
- Examiner: "Why should I trust a 78% model over a 99% one?"
- You: "Because the 99% model uses a feature (`Credit_Score`) that isn't available on the actual unseen applicants it would need to score — it's not usable, and it mostly just re-encodes the bank's own scoring rule rather than learning risk from behavior."
- Examiner: "How do you know it's 'just re-encoding' the rule?"
- You: "I grouped `Credit_Score` by class — P1 is 701–809, P2 is 669–700, P4 is 469–658 — almost no overlap except P3. That's a threshold rule, not a learned pattern."
- Examiner: "Couldn't a real bank still want the 99% model?"
- You: "Only for something like auto-explaining past approval decisions, not for scoring new applicants — the unseen dataset itself excludes that column, so the schema tells you the intended production model shouldn't use it either."

---

## 2.2 Problem Statement

| Question | Short Answer |
|---|---|
| What exactly is the problem? | Given an applicant's bureau/trade-line history, predict which of 4 risk tiers they belong to. |
| Who faces this problem? | Lenders/banks doing loan triage at scale. |
| Why is this problem important? | Manual review doesn't scale; consistent automated tiering reduces bias and turnaround time. |
| Why 4 tiers instead of accept/reject? | Reflects real bank workflow — different tiers get different downstream actions (auto-approve, manual review, etc.), not just approve/deny. |
| What are the consequences of getting it wrong? | Misclassifying a high-risk applicant as low-risk (P4→P1) is costly; misclassifying between adjacent tiers (P2↔P3) is lower-cost — this is why your confusion-matrix "adjacency" finding matters (Section 2.8 below). |

## 2.3 "Existing System" / Related Work (your report's Section II)

| Question | Short Answer |
|---|---|
| What existing approaches did you review? | Classical credit scoring (Logistic Regression, LDA) per Thomas et al.; ensemble benchmarking by Lessmann et al. (2015); XGBoost (Chen & Guestrin 2016); SHAP (Lundberg & Lee 2017); leakage taxonomy (Kaufman et al. 2012). |
| What did Lessmann et al. (2015) find? | Benchmarked 40+ classifiers across 8 real credit datasets — ensembles (RF, boosting) consistently beat single classifiers, at some cost to interpretability. This directly motivated your model choice (LR baseline → RF → XGBoost). |
| What is Kaufman et al.'s leakage taxonomy, and how does your project use it? | They formalize "leaky features" — ones correlated with the label via a business process, not a causal relationship. Your `Credit_Score` finding is a textbook instance of exactly this pattern — you can cite this paper by name to justify *why* you treated it as leakage rather than just "a strong feature." |
| Why didn't you just use an existing published pipeline as-is? | The internal+external merge, the `-99999` sentinel handling, and specifically the leakage discovery were project-specific findings from your actual data — no off-the-shelf pipeline handles a dataset-specific issue like this automatically. |

## 2.4 Proposed System / Methodology

**Full pipeline, in order (know this cold):**
1. Load `internal_bank_dataset.xlsx` (26 features, 51,336 rows) and `external_cibil_dataset.xlsx` (62 features, 51,336 rows, contains target `Approved_Flag`)
2. Clean `-99999` sentinel: drop 40 rows from internal data (`Age_Oldest_TL`); drop 8 sparse columns from external data (threshold: >10,000 sentinel rows), then drop remaining sentinel rows → external data shrinks to 42,066 rows
3. Inner join on `PROSPECTID` → merged dataset: **42,064 rows × 79 columns**
4. EDA: target distribution (P2 = 60.5%), feature distributions by class
5. **Leakage investigation**: found `Credit_Score` almost fully separates classes; confirmed `unseen_dataset.xlsx` excludes it
6. Feature engineering: `EDUCATION` → ordinal encode; `MARITALSTATUS`, `GENDER`, `last_prod_enq2`, `first_prod_enq2` → one-hot encode; target → label encode
7. Define two feature sets: **Model A (Full, 51 features)** vs **Model B (Deployment, 50 features, matches unseen schema)**
8. Train Logistic Regression, Random Forest, XGBoost on both feature sets (80/20 stratified split, `random_state=42`)
9. Evaluate: accuracy + macro-F1, confusion matrices
10. SHAP `TreeExplainer` on Model B (500-sample subset)
11. Score `unseen_dataset.xlsx` with Model B → `unseen_predictions.csv`

| Question | Short Answer |
|---|---|
| Why an inner join and not a left/outer join? | You only want applicants present and clean in *both* sources — a left join would leave nulls for internal-only records, defeating the cleaning work already done. |
| Why clean before merging rather than after? | Cleaning each source independently is simpler and avoids conflating internal vs external missingness; also, internal data had a trivial fix (drop 40 rows) so there was no reason to carry that dirt through the merge. |
| Why `PROSPECTID` as the join key? | It's the shared applicant identifier across both files — the only column common to both by design. |

---

## 2.5 Data Cleaning — `-99999` Sentinel (⚠️ High-value viva topic)

| Question | Short Answer |
|---|---|
| Why not use `NaN`/`.isnull()`? | Because the data doesn't use `NaN` — missingness is encoded as the literal integer `-99999`. `.isnull()` would silently return zero missing values and hide the problem completely. |
| Why drop columns instead of imputing them? | 8 columns were missing in **70%+** of rows (e.g., `CC_utilization` missing in 47,636 of 51,336 rows). Imputing at that scale doesn't recover signal — it fabricates it. A model "learning" from 70%-imputed data is really learning from your imputation strategy, not the applicant. |
| Why the specific threshold of 10,000 rows? | A pragmatic threshold, not a universal rule — chosen after inspecting the missingness distribution and seeing a natural gap between columns with a handful of missing rows and columns missing in the majority of rows. **Be honest if pushed further**: it's a reasonable heuristic, not derived from a formula. |
| What happened to the 40 internal rows? | They had `-99999` in `Age_Oldest_TL`/`Age_Newest_TL` — a tiny fraction (0.08%) of 51,336, so dropping them (vs. imputing) had negligible impact and avoided introducing any artificial values into an important numeric feature. |
| Could dropping rows/columns introduce bias? | Yes — flagged explicitly in the report's "Threats to Validity": if missingness is *not* random (e.g., correlates with true risk), the cleaned sample could be skewed. This wasn't tested but is an honest limitation. |

**Follow-up chain:**
- Examiner: "Why not just impute the 8 dropped columns using KNN or mean imputation instead of dropping them?"
- You: "Because 70–93% of the values were missing — at that scale, imputation isn't recovering real information, it's inventing a majority of the column's values. That signal would be mostly noise, and worse, a model could over-weight it, making it *harder* to trust the model's real drivers."
- Examiner: "What if a smarter imputation method (like MICE) could actually recover useful signal?"
- You: "Possible in principle, but validating that would need its own study — comparing model performance with vs. without imputed columns — which was out of scope here. I made the conservative, defensible choice."

---

## 2.6 The Leakage Discovery (⚠️ Your most important talking point — expect deep questions here)

**What it is:** `Credit_Score` (CIBIL score, range 300–900) is a feature in the external dataset. When included, it lets the model reach 99.2% accuracy — because the bank's own `Approved_Flag` label is almost a direct threshold function of this score.

**Evidence you found (know these numbers exactly):**

| Class | Credit_Score Min | Max | Mean |
|---|---|---|---|
| P1 | 701 | 809 | 716.22 |
| P2 | 669 | 700 | 682.40 |
| P3 | 489 | 776 | 666.82 |
| P4 | 469 | 658 | 645.35 |

Notice: P1, P2, P4 barely overlap. **P3 is the exception** — its range (489–776) spans across all the others, which is *why* P3 is the hardest class to predict later on.

**The confirming clue:** `unseen_dataset.xlsx` — the file meant to simulate real production scoring — has 42 columns and **does not include `Credit_Score` at all**. That's strong evidence the intended, real task deliberately excludes it.

| Question | Short Answer |
|---|---|
| Is this "real" data leakage in the technical sense (future information)? | Not exactly the classic temporal leakage (using future data to predict the past). It's a **feature leakage** / "leaky proxy" pattern per Kaufman et al. — a feature that encodes the label through a business rule rather than being causally predictive on its own. |
| How exactly did you discover it? | Two steps: (1) trained a quick baseline including all features, got a suspiciously high ~99% accuracy; (2) grouped `Credit_Score` by target class and saw near-total separation; (3) cross-checked against `unseen_dataset.xlsx`'s column list and found `Credit_Score` absent. |
| Why is 99% "suspicious" in the first place? | A 4-class problem with imbalanced classes and only bureau/behavioral features rarely hits 99% — that's an instinct built from knowing the difficulty of the underlying prediction task, confirmed by investigation, not assumed. |
| Isn't `Credit_Score` still a genuinely useful real-world feature for a bank? | Yes, if the goal is explaining *why* a past decision was made. But if the goal is predicting new applicants going through the same pipeline that eventually produces `Approved_Flag`, using the bank's own downstream score as an input is circular. |
| What would happen if you'd only trained the 99% model and submitted that? | It would look impressive but be undeployable and scientifically misleading — this is exactly the "leakage misrepresentation" the report calls out explicitly as a risk. |
| Could there be a *second*, subtler leakage source you didn't catch? | Possible — you checked SHAP attributions on Model B and found they're all individually plausible behavioral features (trade-line age, enquiry counts, delinquency level), which is indirect evidence against further leakage, but it isn't a proof. This is an honest limitation to acknowledge if pushed. |

**Follow-up chain (this WILL come up — rehearse it):**
- Examiner: "Why not just remove `Credit_Score` from the start instead of building the leaky model at all?"
- You: "I built it deliberately, not by mistake — to quantify exactly how much accuracy that one column was worth (21.3 points), and to make the tradeoff explicit and defensible in the report rather than silently deciding for the reader."
- Examiner: "Isn't building a model you know is 'wrong' a waste of effort?"
- You: "It's not wrong, it's a different (and simpler) prediction task — modeling the bank's own scoring rule. Comparing it to the honest model is what let me *prove* the leakage claim with numbers rather than just asserting it."

---

## 2.7 Feature Engineering

| Question | Short Answer |
|---|---|
| Why ordinal-encode `EDUCATION` but one-hot encode the rest? | `EDUCATION` has a natural order (SSC < 12TH < Undergraduate < Graduate < Post-Graduate) — ordinal encoding preserves that. `MARITALSTATUS`, `GENDER`, `last_prod_enq2`, `first_prod_enq2` have no inherent order, so one-hot avoids implying a false ranking. |
| ⚠️ Why did `PROFESSIONAL` get the same value (4) as `GRADUATE`, and `OTHERS` the same as `SSC` (1)? | A simplification treating them as roughly equivalent risk/education tiers. **Be ready to admit this is a judgment call, not derived from the data** — a more rigorous approach would check whether `PROFESSIONAL` applicants' actual outcomes justify equating them with `GRADUATE`. |
| Why `drop_first=True` in one-hot encoding? | Avoids the "dummy variable trap" — with `drop_first`, one category becomes the implicit baseline, avoiding perfect multicollinearity among the dummy columns (relevant for Logistic Regression especially). |
| Why label-encode the target instead of keeping it as strings? | Scikit-learn/XGBoost classifiers require numeric targets internally; `LabelEncoder` maps P1–P4 to 0–3 while `target_encoder.classes_` lets you map back to the original labels for reporting. |
| Was feature scaling applied? | **No — this is a real gap.** Logistic Regression was trained on unscaled features (income in the hundreds of thousands vs. binary flags). This likely contributed to LR's weak accuracy (63%) alongside its inherent linearity limitation. Tree-based models (RF, XGBoost) are scale-invariant, so they weren't affected — but if asked directly "did you scale?", the honest answer is no. |

---

## 2.8 Models, Training, and Results

**Model configurations (know these):**

| Model | Key hyperparameters |
|---|---|
| Logistic Regression | `max_iter=1000`, `random_state=42` |
| Random Forest | `n_estimators=200`, `max_depth=10`, `random_state=42` |
| XGBoost | `n_estimators=300`, `max_depth=5`, `learning_rate=0.08`, `subsample=0.9`, `colsample_bytree=0.9`, `min_child_weight=3` |

**Results table (know these numbers exactly — most likely to be quizzed):**

| Feature Set | Model | Accuracy | Macro F1 |
|---|---|---|---|
| Model A (Full, 51 feat.) | Logistic Regression | 0.6435 | 0.3516 |
| Model A | Random Forest | 0.9854 | 0.9742 |
| Model A | **XGBoost** | **0.9924** | **0.9856** |
| Model B (Deployment, 50 feat.) | Logistic Regression | 0.6327 | 0.3507 |
| Model B | Random Forest | 0.7611 | 0.6151 |
| Model B | **XGBoost** | **0.7792** | **0.6879** |

**Model B per-class breakdown (the deployable model — know this table too):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| P1 | 0.83 | 0.78 | 0.80 | 982 |
| P2 | 0.82 | 0.92 | 0.87 | 5,090 |
| P3 | 0.43 | 0.28 | 0.34 | 1,288 |
| P4 | 0.76 | 0.71 | 0.74 | 1,053 |

| Question | Short Answer |
|---|---|
| Why 80/20 split and not something else (e.g., 70/30)? | Standard convention for a dataset of this size (~42k rows) — 20% test (~8,400 rows) is large enough to give stable metrics while leaving 80% for training. |
| Why `stratify=y`? | Preserves the class proportions (P2=60.5% etc.) in both train and test splits — without it, a random split could under- or over-represent minority classes (P1/P3/P4) in the test set, making metrics unreliable. |
| Why `random_state=42`? | For reproducibility — anyone re-running your notebook gets the identical split and identical results. 42 has no special meaning; it's a fixed seed convention. |
| Why did you use **accuracy AND macro-F1**, not just one? | Accuracy alone is misleading under class imbalance — a model predicting "P2" every time gets 60.5% accuracy with zero real skill. Macro-F1 averages F1 *equally* across all 4 classes regardless of size, exposing weak minority-class performance that accuracy hides. |
| Why is Model B's macro-F1 (0.688) so much lower than its accuracy (0.779)? | Because P3's F1 is only 0.34 — macro-F1 doesn't let a big class like P2 (F1=0.87) mask that weakness, while accuracy (dominated by the large P2 class) looks better than the model's real balanced performance. |
| Why is XGBoost the best model in both settings? | Gradient boosting builds trees sequentially, each correcting the previous trees' errors — this generally captures non-linear feature interactions (like `enq_L3m` × `max_recent_level_of_deliq`) better than bagging (RF) or a linear model (LR), consistent with the Lessmann et al. benchmarking literature cited. |
| Why is the RF-vs-XGBoost gap so small (<2 points) while LR trails by ~15–35 points? | RF and XGBoost are both tree-ensemble methods that can model non-linearity — the *ensemble structure* is what matters most here, not the specific boosting mechanism. LR is fundamentally linear and can't capture the same interactions regardless of tuning. |
| Why is P3 the weakest class (F1=0.34, recall=0.28)? | Its `Credit_Score` range (489–776) genuinely overlaps P1, P2, and P4 — meaning even the bank's own scoring policy doesn't cleanly separate P3 applicants. This isn't a modelling failure; it's a property of the label itself. |
| What does the confusion matrix "adjacency" finding mean, and why does it matter? | Model B's errors cluster between *neighboring* tiers (P2↔P3, P3↔P4) rather than distant ones (P1↔P4 is rare). Practically, misclassifying a tier as its neighbor is a lower-cost mistake for a bank's triage workflow than a wild misclassification — this makes the 77.9% number more deployable than it looks in isolation. |
| Did you try to push accuracy higher through more tuning? | Yes — manually varied tree depth, learning rate, and subsampling ratios for XGBoost; accuracy plateaued in a narrow 77–78% band. This suggests the ceiling is **informational** (P3's inherent ambiguity), not a tuning gap. |
| Why not use GridSearchCV/RandomizedSearchCV formally? | Manual exploration was used given time constraints; report explicitly states this as a limitation — a more exhaustive automated search (e.g., via Optuna, listed in Future Work) might find marginal improvement but is unlikely to break the ~78% plateau given the P3 overlap issue. |

**Follow-up chain:**
- Examiner: "Your macro-F1 is only 0.688 — isn't that a weak result?"
- You: "It reflects a genuinely hard 4-class problem where one class (P3) is inherently ambiguous even under the bank's own labeling — but the confusion matrix shows errors are concentrated between adjacent risk tiers, which is a much lower-cost failure mode than random misclassification."
- Examiner: "How would you actually fix P3's performance?"
- You: "Class-balancing techniques like SMOTE for minority classes, or potentially engineering an interaction feature specific to the ambiguous score band — both listed as future work, not yet implemented."
- Examiner: "Why didn't you implement SMOTE now if you knew about it?"
- You: "Time and scope — the priority was correctly identifying and handling the leakage issue first, since that affects the validity of *any* result reported. SMOTE is a refinement on top of an already-honest baseline."

---

## 2.9 SHAP / Interpretability

| Question | Short Answer |
|---|---|
| What is SHAP, in one line? | A game-theory-based method (Shapley values) that fairly attributes a prediction's deviation from the average to each input feature. |
| Why `TreeExplainer` specifically? | It's SHAP's exact, efficient algorithm for tree-based models (RF/XGBoost) — much faster than the general-purpose `KernelExplainer`, which approximates via sampling and works on any model type. |
| Why only 500 samples, not the full test set (8,413 rows)? | Computational cost — SHAP value computation scales with dataset size; 500 is a representative subset sufficient for stable aggregate rankings without excessive runtime. |
| Why explain Model B and not Model A? | Explaining Model A would just re-confirm "`Credit_Score` matters" — already established in Section 7 of the report. Model B is the model that's actually deployable, so its explanation is the one with real value. |
| What does the "3D tensor (samples × features × classes)" mean, practically? | For a 4-class problem, SHAP gives a separate attribution value for *each class* per sample per feature — i.e., how much each feature pushed the prediction toward P1 vs P2 vs P3 vs P4 individually. You averaged the absolute value across the class axis to get one global importance ranking. |
| What are the actual top features (know this — verified against the real chart)? | In order: `Age_Oldest_TL` (trade-line age), `enq_L3m` (recent enquiries), `max_recent_level_of_deliq` (recent delinquency severity), `num_std_12mts`, `time_since_recent_enq`. |
| Why does the ranking change for class P4 specifically? | For P4 (highest risk), `enq_L3m` and `max_recent_level_of_deliq` overtake `Age_Oldest_TL` as the top drivers — meaning *recent* enquiry/delinquency behavior matters more for identifying the worst-risk applicants specifically, while account longevity matters more for the overall/aggregate ranking. |
| Why does SHAP interpretability matter for a bank use case? | Regulatory requirement in lending — rejected applicants can, in principle, be shown which specific factors drove their classification, supporting auditability and fairness reviews. |
| What's the difference between SHAP and simple feature importance (e.g., `feature_importances_` from RF)? | Built-in feature importance (e.g., Gini importance) is a *global*, model-internal measure with known biases (favors high-cardinality features). SHAP is *per-prediction*, theoretically grounded in cooperative game theory, and gives both local (single-applicant) and global (aggregated) explanations. |

**Follow-up chain:**
- Examiner: "How do you know SHAP's top features aren't just noise from your 500-sample subset?"
- You: "500 is a large enough sample to stabilize aggregate rankings for a dataset this size, and the top features align with domain intuition — trade-line age and recent enquiry behavior are exactly what a credit analyst would expect to matter. That domain-plausibility is itself a sanity check."
- Examiner: "Could you have used the full 8,413-row test set instead?"
- You: "Yes, at higher computational cost — for a course project, 500 was a reasonable time/coverage tradeoff, not a hard constraint."

---

## 2.10 Scoring the Unseen Dataset

| Question | Short Answer |
|---|---|
| What is `unseen_dataset.xlsx`? | 100 applicants, 42 columns, **no target label, no `Credit_Score`** — simulates true production scoring. |
| Why does this file matter methodologically? | It's the evidence that confirmed Model B's feature set (no `Credit_Score`) is the intended, real-world-usable one — Model A literally cannot be applied to it. |
| How did you align the unseen data's columns with the trained model's expected features? | Applied the same `EDUCATION` mapping and one-hot encoding used in training, then used `df.reindex(columns=DEPLOYMENT_FEATURES, fill_value=0)` to guarantee exact column match/order, adding any missing dummy columns as 0. |
| What did you do with the predictions? | Saved predicted class + confidence (max class probability) per applicant to `unseen_predictions.csv`. |
| How would you validate these predictions are trustworthy, given there's no ground truth for the unseen set? | You can't directly — this is an honest limitation. The only evidence of trustworthiness is Model B's performance on the *labeled* held-out test set (77.9% accuracy), which is why that number, not the unseen predictions themselves, is the headline result. |

---

# 3. ⚠️ Technical Terms — VIVA RISK Flags

For each: **simple explanation + safe viva answer.**

### 🔴 Data Leakage
**Simple:** A feature that "cheats" by encoding the answer instead of genuinely predicting it.
**Safe answer:** "A leaky feature is one that correlates with the target not because it causally drives risk, but because it's part of the same business process that produced the label — like `Credit_Score` being the input to the bank's own approval rule."

### 🔴 SHAP / Shapley Values
**Simple:** A fair way to split credit for a prediction among all the features that contributed to it, borrowed from cooperative game theory (how to fairly split a group's payoff among members based on their contribution).
**Safe answer:** "SHAP assigns each feature a numeric contribution to how far a prediction moved from the average prediction, computed via Shapley values from game theory — guaranteeing a mathematically fair, consistent attribution."

### 🟠 Macro-F1
**Simple:** Average the F1 score of each class equally, ignoring how many samples each class has.
**Safe answer:** "Macro-F1 computes F1 per class then averages unweighted — so a class with 1,000 samples counts the same as a class with 10,000. This stops a large class from hiding poor performance on small classes."

### 🟠 Gradient Boosting (XGBoost mechanism)
**Simple:** Build one small tree, see what it got wrong, build the next tree to fix those specific mistakes, repeat.
**Safe answer:** "Each new tree is trained on the residual errors of the ensemble so far, using gradient descent on a loss function — so the model incrementally corrects itself, unlike Random Forest where trees are independent."

### 🟠 Bagging vs. Boosting
**Simple:** Bagging = many independent trees vote together (Random Forest). Boosting = trees built one after another, each fixing the last one's mistakes (XGBoost).
**Safe answer:** "Bagging reduces variance by averaging independent models trained on bootstrapped samples. Boosting reduces bias by sequentially focusing each new model on the previous ensemble's errors."

### 🟠 One-Hot vs. Ordinal Encoding
**Simple:** One-hot = separate yes/no column per category (no order implied). Ordinal = one column with ranked numbers (order implied).
**Safe answer:** "Ordinal encoding is appropriate when categories have a genuine order, like education level. One-hot avoids implying a false numeric order for unordered categories like gender or marital status."

### 🟡 Stratified Split
**Simple:** Splitting train/test while keeping the same class proportions in both.
**Safe answer:** "`train_test_split(..., stratify=y)` ensures both splits have the same ~60.5%/15.3%/12.5%/11.7% class ratio as the full dataset, so metrics on the test set are representative."

### 🟡 `TreeExplainer` vs. `KernelExplainer`
**Simple:** `TreeExplainer` is a fast shortcut that only works for tree models. `KernelExplainer` is slower but works for any model.
**Safe answer:** "`TreeExplainer` exploits the known structure of decision trees to compute exact SHAP values efficiently; it's specific to tree-based models like XGBoost/RF, unlike the model-agnostic but much slower `KernelExplainer`."

### 🟡 Multiclass classification internals (XGBoost)
**Simple:** Instead of yes/no, the model outputs a probability for each of the 4 classes and picks the highest.
**Safe answer:** "XGBoost's multiclass objective (`mlogloss`) internally uses a softmax-style approach — it produces one probability per class that sums to 1, and predicts the class with highest probability, rather than training 4 separate one-vs-rest models."

### 🟢 `PROSPECTID`
**Simple:** Just the unique applicant ID shared by both datasets — the join key.

### 🟢 CIBIL
**Simple:** Credit Information Bureau (India) Limited — India's credit bureau, the source of the external dataset.

---

# 4. Technology Questions (What / Why / How / Alternatives)

| Technology | What | Why used | Key alternative | Why not the alternative |
|---|---|---|---|---|
| **Pandas** | DataFrame library for tabular data manipulation | Standard for cleaning/merging/feature engineering at this scale (~50k rows) | Polars, Dask | Overkill — dataset fits comfortably in memory; Pandas has a simpler, more familiar API for this scope |
| **XGBoost** | Gradient-boosted tree library | Best empirical performance in both feature settings; industry-standard for tabular data competitions | LightGBM, CatBoost | Not tested — XGBoost was sufficient and is the most widely documented/supported for a course project; LightGBM could be faster on larger data, CatBoost handles categoricals natively (a possible future-work angle) |
| **Scikit-learn** | ML utilities: `train_test_split`, `LabelEncoder`, `LogisticRegression`, `RandomForestClassifier`, metrics | Standard, well-tested implementations; consistent API across all three classifiers | Statsmodels (for LR specifically) | Statsmodels gives richer statistical inference (p-values, confidence intervals) but scikit-learn's simpler `.fit()/.predict()` API is more consistent with the RF/XGBoost workflow used alongside it |
| **SHAP** | Model interpretability library | Only well-supported library offering exact, efficient explanations for tree ensembles, with strong multiclass support | LIME | LIME approximates locally via perturbation and is generally considered less theoretically grounded and less stable than SHAP's Shapley-value guarantees |
| **Matplotlib / Seaborn** | Plotting | Standard visualization stack, sufficient for static report figures | Plotly | Interactive plots aren't needed for a static PDF report |
| **Joblib** | Model serialization | Standard for persisting scikit-learn/XGBoost-compatible models to `.pkl` | Pickle directly | Joblib is more efficient for objects containing large NumPy arrays (like fitted trees) |
| **openpyxl** | Reading `.xlsx` files via Pandas | Required engine for Excel I/O since source files are `.xlsx` | — | Not really an alternative-worthy choice — dictated by the input file format |

**Follow-up chain:**
- Examiner: "Why XGBoost over LightGBM specifically?"
- You: "XGBoost is more heavily documented and had proven reliable results here; LightGBM is generally faster on very large datasets via histogram-based splitting, but our dataset (~42k rows) isn't large enough for that advantage to matter much."
- Examiner: "What if the dataset were 10x bigger — would your choice change?"
- You: "Possibly — at that scale I'd benchmark LightGBM's training-time advantage, since XGBoost can get slow on very large data without careful tuning."

---

# 5. Algorithm Deep-Dives

## 5.1 Logistic Regression (baseline)
- **Core principle:** Models the log-odds of each class as a linear combination of features; for multiclass, scikit-learn uses a multinomial/softmax extension.
- **Why included:** Interpretable baseline — establishes a performance floor and shows how much non-linearity actually matters for this problem.
- **Result:** 63–64% accuracy in both feature settings — far behind the ensembles.
- **Why it underperforms here:** (1) genuinely non-linear feature interactions it can't capture; (2) **no feature scaling was applied**, which likely hurts convergence/coefficient stability given wildly different feature scales (income vs. binary flags).
- **Limitation to admit:** You didn't isolate how much of LR's weak performance is scaling vs. linearity — an honest "I'd need to re-run with `StandardScaler` to separate those effects" is a good answer if pushed.

## 5.2 Random Forest
- **Core principle:** Bagging — train many decision trees independently on bootstrapped samples with random feature subsets at each split, then average/vote.
- **Key hyperparameters used:** `n_estimators=200`, `max_depth=10`.
- **Why `max_depth=10`:** Caps tree complexity to reduce overfitting risk given ~34k training rows and 50–51 features.
- **Result:** 76.1–98.5% depending on feature set — strong, close to XGBoost.
- **Advantage over LR:** Captures non-linear interactions and feature thresholds naturally.
- **Disadvantage vs. XGBoost:** No error-correction mechanism between trees — each tree is blind to others' mistakes, generally capping its ceiling slightly below boosting.

## 5.3 XGBoost (final/best model)
- **Core principle:** Gradient boosting — sequentially add trees, each fitting the *gradient of the loss* with respect to the current ensemble's predictions (i.e., correcting residual errors), with L1/L2 regularization to control overfitting.
- **Key hyperparameters used:** `n_estimators=300`, `max_depth=5` (shallower than RF's 10 — boosting needs shallower trees since errors compound across many rounds), `learning_rate=0.08` (how much each tree's correction is trusted), `subsample=0.9`, `colsample_bytree=0.9` (both add randomness to reduce overfitting, similar in spirit to RF's bagging), `min_child_weight=3` (minimum samples-weight needed to split further, another overfitting guard).
- **Why shallower trees than RF (`max_depth=5` vs `10`)?** In boosting, errors compound across hundreds of sequential trees — deep trees would overfit fast; shallow "weak learners" combined many times over is the core boosting philosophy.
- **Result:** Best in both settings — 99.24% (Model A), 77.92% (Model B).
- **Computational note:** Not deeply profiled in this project; runtime was acceptable for the dataset scale (tens of thousands of rows).
- **Failure case exposed:** Even XGBoost tops out at ~78% on Model B because of P3's inherent label ambiguity — the algorithm isn't the bottleneck at that point, the label overlap is.

## 5.4 SHAP TreeExplainer
- **Core principle:** Computes exact Shapley values efficiently for tree ensembles by exploiting the tree structure (rather than the exponential brute-force computation Shapley values would otherwise require).
- **Input:** Trained tree model + a sample of feature rows.
- **Output:** Per-sample, per-feature, per-class attribution values (here: a 500×50×4 tensor).
- **Aggregation used:** Mean absolute value across the class axis, then across samples, to get one global ranking.

---

# 6. "How Does the Whole System Work?" — Full Workflow Walkthrough

Practice narrating this without looking at the notebook:

1. **Input:** Two Excel files (`internal_bank_dataset.xlsx`, `external_cibil_dataset.xlsx`) loaded via `pd.read_excel()`.
2. **Cleaning:** Sentinel `-99999` values identified and removed (rows from internal, columns+rows from external) — see Section 2.5 above for exact numbers.
3. **Merge:** `pd.merge(df_internal, df_external, how='inner', on='PROSPECTID')` → single 42,064-row, 79-column DataFrame.
4. **EDA:** Class distribution and feature-distribution plots generated to understand the data before modelling.
5. **Leakage check:** `Credit_Score` grouped by `Approved_Flag`; boxplot and summary table produced; cross-checked against `unseen_dataset.xlsx` columns.
6. **Feature engineering:** `EDUCATION` ordinal-mapped via a Python dict; four other categoricals one-hot encoded via `pd.get_dummies(..., drop_first=True)`; target label-encoded via `LabelEncoder`.
7. **Feature set definition:** `DEPLOYMENT_FEATURES` (50 cols, matches unseen schema) vs `FULL_FEATURES` (51 cols, adds `Credit_Score`).
8. **Split:** `train_test_split(..., test_size=0.2, random_state=42, stratify=y)` applied separately to each feature set (same random state → same row split).
9. **Train:** Three classifiers fit on each of the two feature sets (6 total model fits) via a reusable `train_and_evaluate()` function.
10. **Evaluate:** Accuracy, macro-F1, classification report, confusion matrix computed per model; results compared visually (bar charts) and numerically (tables).
11. **Persist:** Best model (XGBoost) from each feature set saved via `joblib.dump()` → `model_full_with_credit_score.pkl`, `model_deployment.pkl`.
12. **Explain:** SHAP `TreeExplainer` run on Model B; global and P4-class-specific attribution plots generated.
13. **Score unseen data:** `unseen_dataset.xlsx` loaded, same encoding applied, columns reindexed to match `DEPLOYMENT_FEATURES` exactly, `model_deployment.pkl` used to predict class + confidence per applicant, saved to `unseen_predictions.csv`.

**What happens if invalid input reaches the pipeline?** Not explicitly handled — there's no input validation layer (e.g., checking for out-of-range values or unexpected categories in new data) beyond the sentinel-cleaning step, which only applies to the two training source files. **Honest answer if asked:** "Input validation for arbitrary new data wasn't built — the unseen-scoring step assumes the same schema and encoding conventions as the training data, via the `reindex(fill_value=0)` step, which would silently zero-fill any genuinely new/unexpected category rather than raising an error." This is a real limitation worth acknowledging.

---

# 7. Verbal "Architecture Diagram" (since none exists — use this structure if asked to draw one on the whiteboard)

```
[internal_bank_dataset.xlsx]     [external_cibil_dataset.xlsx]
         |                                |
   drop 40 rows (-99999)          drop 8 sparse cols,
         |                        then drop -99999 rows
         |                                |
         +---------- inner join on PROSPECTID ----------+
                              |
                   Merged dataset (42,064 x 79)
                              |
                    EDA + Leakage Investigation
                     (Credit_Score vs target)
                              |
                    Feature Engineering
              (ordinal EDUCATION, one-hot others)
                              |
              +---------------+---------------+
              |                               |
     FULL_FEATURES (51)              DEPLOYMENT_FEATURES (50)
      [Model A]                           [Model B]
              |                               |
     LR / RF / XGBoost                LR / RF / XGBoost
        (train + eval)                  (train + eval)
              |                               |
      99.2% acc (not deployable)      77.9% acc (deployable)
                                              |
                                   SHAP explainability
                                              |
                              Score unseen_dataset.xlsx
                                              |
                              unseen_predictions.csv
```

---

# 8. Results / Tables / Graphs Questions

For each figure/table in the report:

**Table I (Target distribution):** "Shows the class imbalance that motivates using macro-F1. P2 alone is 60.5% — a naive always-predict-P2 model would already hit that accuracy."

**Table II (Credit_Score by class) / Fig 2 (boxplot):** "Shows the leakage evidence numerically and visually — near-total separation for P1/P2/P4, wide overlap for P3."

**Table III (Model comparison) / Fig 3:** "Six model×feature-set combinations. Shows XGBoost wins both settings, and quantifies the leakage cost as a 21.3-point accuracy gap."

**Table IV (Model B per-class metrics):** "The real, deployable model's honest per-class breakdown — used to identify P3 as the weak point, not just report one blended number."

**Fig 4 (confusion matrices):** "Model A (left) is near-diagonal because of the Credit_Score shortcut; Model B (right) shows adjacency — errors cluster between neighboring tiers."

**Fig 5 (SHAP summary):** "Global feature importance for Model B, averaged across all 4 classes — validates the model uses plausible behavioral drivers, not something uninterpretable."

**Generic follow-ups for any of the above:**
- "How was this number obtained?" → point to the specific notebook cell/function (`train_and_evaluate()`, `groupby().agg()`, etc.)
- "Is this good or bad?" → always benchmark against the naive baseline (predict majority class = ~60.5% accuracy) and against published literature (75–80% typical for comparable 4-class bureau datasets).
- "Could this change under different conditions?" → yes: different random seed → slightly different split → metrics could shift by a percentage point or two; not tested via cross-validation, which is an acknowledged limitation.

---

# 9. Testing & Evaluation — Be Honest About Scope

**What was actually done:**
- A single 80/20 stratified train/test split (not k-fold cross-validation)
- Standard classification metrics: accuracy, macro-F1, per-class precision/recall/F1, confusion matrices
- A manual sanity check that SHAP's tensor shape and the unseen dataset's column alignment were correct before scoring

**What was NOT done (say this plainly if asked, don't invent):**
- No k-fold cross-validation → no variance/confidence interval on the reported metrics
- No formal unit tests on the code itself
- No independent validation set held out from hyperparameter selection (the same test set was effectively used to compare configurations during manual tuning — a mild form of test-set leakage in the *tuning process*, worth admitting if pressed)

**If asked "how do you know your system works correctly?"**
"I verified correctness at each pipeline stage rather than through formal test cases — e.g., confirming the merged row/column counts matched expectations after cleaning, confirming the SHAP tensor shape matched (samples, features, classes), and confirming `unseen_dataset.xlsx`'s columns fully aligned with `DEPLOYMENT_FEATURES` before scoring, with zero unaccounted columns. That's stage-by-stage validation, not automated testing."

---

# 10. Top Challenging / Trick Questions

| Question | Strong Answer |
|---|---|
| What's the single biggest limitation of your project? | The lack of k-fold cross-validation means the reported 77.9% is a point estimate from one split — I don't have a confidence interval on it. |
| What's the weakest part of your implementation? | P3 classification (F1=0.34) — even the best model struggles because the label itself is ambiguous for that class, not purely a modelling shortfall. |
| Why should I believe your 77.9% number? | It's on a held-out stratified test set the model never saw during training, using an honest feature set that matches the schema the project's own unseen-scoring file expects — and it's consistent with published benchmarks for similar bureau-based 4-class problems. |
| What's the biggest assumption in your project? | That the `-99999` sentinel handling (dropping columns >10k missing, dropping remaining sentinel rows) doesn't introduce meaningful selection bias — this wasn't tested, only reasoned about. |
| What could cause your system to fail on new data? | A genuinely new applicant with a category value not seen in training (e.g., a new `loan_intent`-style category) would silently get zero-filled by the `reindex` step rather than raising an error — a real robustness gap. |
| If you had 3 more months, what would you change? | Add k-fold CV for confidence intervals, apply SMOTE for P3/P1/P4 minority classes, automate hyperparameter search via Optuna, and formally test whether feature scaling improves Logistic Regression. |
| What is something your project cannot currently do? | It cannot explain *why* P3 specifically is hard beyond the score-overlap observation — a deeper feature-interaction analysis specific to P3 misclassifications wasn't performed. |
| If I forced you to remove SHAP entirely, what would you lose? | The ability to verify the deployment-safe model relies on plausible behavioral signals rather than some other hidden shortcut — SHAP was your main defense against a "second leakage channel" existing. |
| What's the most important design decision you made? | Building and reporting two models instead of hiding the leakage and presenting only the 99% number. |
| What decision would you reconsider now? | The manual grid search for XGBoost tuning — a proper `RandomizedSearchCV`/Optuna run would give more confidence the 77–78% is a genuine ceiling rather than a local optimum from limited manual exploration. |

---

# 11. Must Memorize vs. Must Understand

### A. Must Memorize (exact facts — say these instantly, no hesitation)
- Project objective: predict 4-tier loan approval priority using internal + external bureau data
- Datasets: 51,336 rows each, internal=26 features, external=62 features; merged = 42,064 rows × 79 columns
- Target imbalance: P2 = 60.51%
- Leakage feature: `Credit_Score`
- Two models: Model A (Full, 51 features, 99.24% acc) vs Model B (Deployment, 50 features, 77.92% acc)
- Best algorithm: XGBoost, in both settings
- Model B macro-F1: 0.688
- Weakest class: P3 (F1 = 0.34)
- Tech stack: Pandas, Scikit-learn, XGBoost, SHAP, Joblib
- Unseen dataset: 100 applicants, 42 features, no target, no `Credit_Score`

### B. Must Understand (never memorize word-for-word — be ready to explain in your own words)
- *Why* `Credit_Score` constitutes leakage (not just "it's a strong feature")
- *How* SHAP's TreeExplainer works conceptually (game-theory attribution via tree structure)
- *How* boosting differs from bagging, and why that explains XGBoost > RF > LR
- *Why* macro-F1 was necessary given the class imbalance
- *Why* the confusion-matrix adjacency pattern matters for real deployment
- *What* the `-99999` cleaning thresholds mean and why they're heuristics, not formulas
- The full pipeline flow end-to-end (Section 6/7 above) — practice narrating without notes

---

# 12. Top Viva Risk Areas — Ranked

### 🔴 High Risk
1. **"Explain exactly why `Credit_Score` is leakage vs. just being a strong predictor."** — Rehearse the Section 2.6 explanation until it's automatic. This is your report's central claim; a shaky answer here undermines everything else.
2. **"Why didn't you scale features for Logistic Regression?"** — Admit the gap honestly (Section 2.7); don't pretend it was intentional.
3. **"Justify why 78% is a good result, not a failure."** — Use the naive-baseline comparison (60.5%) + published-benchmark comparison (75-80% typical) + the P3 label-ambiguity explanation together.
4. **"Walk me through the full pipeline without the notebook open."** — Practice Section 6 until fluent.

### 🟠 Medium Risk
5. **"Why the specific 10,000-row missingness threshold for dropping columns?"** — Be ready to call it a pragmatic heuristic, not a formula.
6. **"Explain SHAP's multiclass tensor and how you aggregated it."** — Know the (samples, features, classes) shape and the mean-absolute-across-classes aggregation.
7. **"Why is P3 specifically the hardest class?"** — Tie directly back to the Credit_Score range overlap (489–776 spans P1/P2/P4's ranges).

### 🟢 Low Risk
8. Technology choice justifications (Pandas, XGBoost, SHAP) — standard, well-documented reasoning, unlikely to be probed deeply.
9. Basic definitions (precision/recall/F1, train-test split) — foundational ML knowledge, low probability of a truly hard follow-up.

---

# 13. Final Preparation Lists

## A. Top 50 Most Likely Questions (ranked, most → least likely)
1. What is your project about?
2. What is `Approved_Flag` / what do P1–P4 mean?
3. What is data leakage and how did you find it in your project?
4. Why does `Credit_Score` cause leakage?
5. What are Model A and Model B, and why do both exist?
6. What's your final accuracy and macro-F1?
7. Why macro-F1 and not just accuracy?
8. Why is the dataset imbalanced, and how much?
9. What algorithm performed best, and why?
10. Explain your full data cleaning process (`-99999`).
11. Why drop columns instead of imputing them?
12. What's the difference between the internal and external datasets?
13. What is `PROSPECTID` and why is it used to merge?
14. Why did you use an inner join?
15. What is SHAP and why did you use it?
16. What are the top SHAP features for your model?
17. Why is class P3 the hardest to predict?
18. What does the confusion matrix show for Model B?
19. What is the "adjacency" property you found in the confusion matrix?
20. Why didn't you scale features for Logistic Regression?
21. Why did you choose XGBoost's specific hyperparameters?
22. What's the difference between bagging and boosting?
23. Why use a stratified train-test split?
24. Why `random_state=42`?
25. What is the unseen dataset, and what did you do with it?
26. How did you align the unseen dataset's columns with your model?
27. Why does the unseen dataset not have `Credit_Score`?
28. What would happen if you deployed Model A instead of Model B?
29. What's your biggest limitation?
30. Why didn't you use k-fold cross-validation?
31. What is `TreeExplainer` and why not `KernelExplainer`?
32. Why is Logistic Regression's accuracy so much lower than the ensembles?
33. What does "macro-F1 of 0.688" actually tell you that accuracy doesn't?
34. Why did you one-hot encode some features and ordinal-encode others?
35. Why did `PROFESSIONAL` and `OTHERS` get merged into other education tiers?
36. What's the difference between Model A's and Model B's confusion matrices?
37. How would you improve P3's classification performance?
38. What future work would you prioritize?
39. Why is this a 4-class problem instead of binary default prediction?
40. What does "deployment-safe" actually mean in your report?
41. Could there be a second hidden leakage feature you missed?
42. What's the computational cost of SHAP, and why sample only 500 rows?
43. What are the exact numbers in your Credit_Score-by-class table?
44. Why is the RF-vs-XGBoost gap so small compared to LR's gap?
45. What tools/libraries did you use, and why each one?
46. What happens if a new applicant has a category not seen in training?
47. How did you save/persist your trained models?
48. What testing did you perform to verify your pipeline's correctness?
49. What's the business cost difference between a P1→P4 misclassification and a P2→P3 misclassification?
50. If you had more time, what's the first thing you'd add?

## B. Top 25 Must-Never-Get-Wrong
1. What P1–P4 mean (best → worst risk)
2. Why `Credit_Score` is leakage
3. Model A vs Model B — which is deployable and why
4. Final headline number: 77.9% accuracy / 0.688 macro-F1 (Model B, XGBoost)
5. Best algorithm across both settings: XGBoost
6. Why macro-F1 matters given imbalance
7. Merged dataset size: 42,064 rows × 79 columns
8. What `-99999` represents and why it's handled specially
9. Why P3 is the weakest class
10. What SHAP is, at a conceptual level
11. Confusion matrix adjacency finding
12. Why an inner join on `PROSPECTID`
13. What the unseen dataset is and why it excludes `Credit_Score`
14. Full pipeline order, start to finish
15. Why stratified split
16. Why LR underperforms
17. Difference between bagging (RF) and boosting (XGBoost)
18. That feature scaling was NOT applied (own this gap)
19. That formal cross-validation was NOT performed (own this gap)
20. Tech stack: Pandas, Scikit-learn, XGBoost, SHAP, Joblib
21. Class imbalance number: P2 = 60.51%
22. The 21.3-point accuracy gap between Model A and B, and what it represents
23. That this is a notebook-based analysis project, not a deployed app (don't invent architecture)
24. Top SHAP features (Age_Oldest_TL, enq_L3m, max_recent_level_of_deliq, ...)
25. What "genuinely deployable" means in context (matches unseen schema)

## C. Top 20 Difficult Questions
1. How exactly does SHAP's Shapley-value computation work mathematically (conceptually, not full derivation)?
2. Why is XGBoost's `max_depth=5` shallower than Random Forest's `max_depth=10` — what does that reveal about boosting vs. bagging philosophy?
3. Could a second, subtler leakage feature exist beyond `Credit_Score`? How would you check?
4. Is the "18% discarded during cleaning" attributable to the join or to the cleaning steps? (Answer: mostly cleaning, not the join itself)
5. What's the practical difference between a feature being "leaky" (Kaufman et al. sense) vs. simply being "very predictive"?
6. Why does removing `Credit_Score` cost exactly 21.3 accuracy points and not more/less?
7. If P3 is inherently ambiguous even in the ground truth, is there a theoretical ceiling on any model's P3 performance? What would that ceiling be?
8. How would you formally test whether your `-99999` cleaning thresholds introduced bias?
9. Why didn't your manual XGBoost tuning explore regularization parameters (`reg_alpha`/`reg_lambda`) explicitly?
10. What's the mathematical difference between Gini importance (used by RF) and SHAP importance?
11. How would multicollinearity among your 50 features affect Logistic Regression specifically (vs. the tree models)?
12. Why is `min_child_weight=3` used, and what does it prevent?
13. What would a k-fold cross-validated confidence interval likely look like for your 77.9% figure, qualitatively?
14. How does XGBoost's `mlogloss` objective mathematically differ from binary logloss?
15. Why is average SHAP magnitude (not raw signed SHAP) the right aggregation for global importance?
16. Could label noise in `Approved_Flag` itself (not just P3 ambiguity) be limiting your ceiling?
17. What's the risk of using the same test set for both final evaluation and manual hyperparameter comparison?
18. How would your leakage argument change if `Credit_Score`'s class overlap were larger (e.g., if P3 overlap extended to all 4 classes)?
19. Why didn't one-hot encoding `last_prod_enq2`/`first_prod_enq2` (potentially high-cardinality) risk a dimensionality blow-up?
20. If the bank's real approval policy changes tomorrow, would your model need retraining? Why?

## D. Top 20 "Why?" Questions
1. Why build two models instead of one?
2. Why treat `Credit_Score` as leakage instead of just a feature?
3. Why macro-F1 alongside accuracy?
4. Why XGBoost over Random Forest as the final model?
5. Why an inner join on `PROSPECTID`?
6. Why drop sparse columns instead of imputing?
7. Why ordinal-encode `EDUCATION` but one-hot the rest?
8. Why `drop_first=True` in one-hot encoding?
9. Why `stratify=y` in the train-test split?
10. Why `random_state=42`?
11. Why 80/20 and not another split ratio?
12. Why explain Model B with SHAP and not Model A?
13. Why only 500 samples for SHAP computation?
14. Why is `max_depth` different between RF (10) and XGBoost (5)?
15. Why didn't you scale features for Logistic Regression?
16. Why didn't you use k-fold cross-validation?
17. Why is this framed as a 4-class problem, not binary?
18. Why does the unseen dataset matter to your methodology?
19. Why is P3 excluded from being "cleanly" solvable, in your view?
20. Why is a 78% model preferable to a 99% model in this context?

## E. Top 20 "How?" Questions
1. How did you discover the `Credit_Score` leakage?
2. How does XGBoost's gradient boosting work step by step?
3. How does Random Forest differ mechanically from XGBoost?
4. How does SHAP's TreeExplainer compute attributions efficiently?
5. How did you clean the `-99999` sentinel values?
6. How did you merge the internal and external datasets?
7. How did you encode categorical features?
8. How did you evaluate model performance across 6 model×feature-set combinations?
9. How did you align the unseen dataset's schema with your trained model?
10. How did you save and reuse your trained models?
11. How does the confusion matrix reveal the "adjacency" pattern?
12. How would you explain a single applicant's prediction using SHAP?
13. How does class imbalance affect accuracy vs. macro-F1 differently?
14. How did you decide the 10,000-row column-drop threshold?
15. How does Logistic Regression handle 4 classes internally (multinomial)?
16. How would you add SMOTE to this pipeline if asked to?
17. How would cross-validation change your evaluation setup?
18. How does one-hot encoding avoid implying false ordinal relationships?
19. How did you verify your unseen-scoring step was correct before trusting it?
20. How would you explain this entire project to a non-technical bank manager?

## F. Top 20 "What If?" Questions
1. What if `Credit_Score` had less overlap between P1/P2/P4 — would your leakage claim be different?
2. What if the dataset were 10x larger — would your model/technology choices change?
3. What if a new applicant has a category value never seen in training?
4. What if you'd used a left join instead of inner join?
5. What if you had imputed the 8 dropped columns instead of dropping them?
6. What if the class imbalance were even more extreme (e.g., P2 = 90%)?
7. What if XGBoost weren't available — what would you use instead?
8. What if the bank's approval policy changes after deployment?
9. What if you removed SHAP entirely from the project?
10. What if you only had the internal dataset, not the external one?
11. What if `Credit_Score` were provided in the unseen dataset after all?
12. What if your test set metrics don't generalize to truly new applicants?
13. What if P3 could never be solved above 40% F1 no matter the model?
14. What if the examiner asked you to explain this project with zero jargon?
15. What if you had to support real-time scoring instead of a batch notebook?
16. What if a stakeholder demanded you report only the 99% model?
17. What if your one-hot encoded categorical columns had 100+ categories?
18. What if you needed to explain a single rejected applicant's decision to them directly?
19. What if you discovered a bug in your `-99999` cleaning logic after submitting the report?
20. What if the examiner asked you to justify every single hyperparameter value?

## G. Top 20 Technology Questions
(See Section 4 table above for full What/Why/How/Alternatives — quick-fire versions:)
1. What is Pandas and why used?
2. What is XGBoost and why used over alternatives?
3. What is Scikit-learn used for here?
4. What is SHAP and why used over LIME?
5. Why Matplotlib/Seaborn and not Plotly?
6. What is Joblib and why not raw Pickle?
7. What is `openpyxl` and why needed?
8. Why Jupyter Notebook as the development environment?
9. What Python version/environment was used? *(Not explicitly pinned beyond `requirements.txt` — check your own setup if asked)*
10. Why not use a deep learning approach (neural network) instead?
11. Why not use CatBoost, given native categorical support?
12. Why not use LightGBM for speed?
13. What does `verbosity=0` do in the XGBoost config?
14. What does `eval_metric='mlogloss'` mean?
15. Why `n_jobs=-1` in Random Forest?
16. What's the role of `LabelEncoder` vs `OneHotEncoder` in this project?
17. Why is `classification_report`'s `output_dict=True` used?
18. What does `ConfusionMatrixDisplay.from_predictions` do?
19. Why is `ColumnTransformer`/pipelines NOT used here (manual encoding instead)?
20. What would change if this were rebuilt using PyTorch/TensorFlow?

## H. Top 20 Results/Evaluation Questions
1. What's your final accuracy and macro-F1?
2. What's the accuracy gap between Model A and Model B, and why?
3. What's Model B's precision/recall/F1 for each class?
4. Which class has the lowest recall, and why?
5. Why is P2's F1 (0.87) so much higher than P3's (0.34)?
6. What does "support" mean in your classification report?
7. How many test samples were there in total (8,413)?
8. What's the naive baseline accuracy, and how does your model compare?
9. Is 77.9% considered good for this type of problem?
10. How consistent were your results across different XGBoost hyperparameter configurations?
11. What would a confusion matrix look like for a "perfect" model, and how far is yours from that?
12. Why didn't Random Forest and XGBoost differ by much in Model A but differ slightly more in Model B?
13. What's the real-world cost interpretation of your confusion matrix's adjacency pattern?
14. How would your results change if evaluated via 5-fold cross-validation instead?
15. What metric would you prioritize if false negatives (missing a risky applicant) were costlier than false positives?
16. Why didn't you report ROC-AUC for this multiclass problem?
17. How do you know your evaluation isn't overfit to the specific train/test split?
18. What does a macro-F1 of exactly 0.688 mean in plain English?
19. If you retrained with a different random seed, would 77.9% likely hold?
20. What's the single most important number in your entire results section, and why?

## I. Top 20 Questions From Figures/Tables/Diagrams
1. Walk me through Table I — what does it show and why include it?
2. Walk me through Table II — how does it support your leakage claim?
3. Walk me through Fig 2 (boxplot) — what does the P3 overlap visually tell you?
4. Walk me through Table III — why compare 6 combinations instead of just the best one?
5. Walk me through Fig 3 — what's the visual takeaway?
6. Walk me through Table IV — why break down by class instead of one blended number?
7. Walk me through Fig 4 — what's the difference between the two confusion matrices?
8. What does "adjacency" mean in Fig 4, concretely?
9. Walk me through Fig 5 — what does the bar length represent?
10. Why does Fig 5 only show ~15 features, not all 50?
11. Is Fig 1's target distribution chart necessary, or redundant with Table I?
12. Why weren't more figures included (e.g., a system architecture diagram)?
13. Could you generate a P4-specific SHAP plot? What would it show? *(Yes — you have one in your notebook's `plots/` folder, `07_shap_p4_class.png`, though it's not in the final report due to page constraints)*
14. Why is there no ROC curve in your report?
15. Why is there no learning curve (train vs. test error over iterations)?
16. If you had one more figure to add, what would it be and why?
17. What conclusion can you legitimately draw from Fig 4 alone?
18. Could Table II's numbers change if you used a different random seed? Why/why not? *(No — Table II is a data-level summary, not model-dependent, so it wouldn't change with a random seed)*
19. What's the difference between Table III's numbers and what you'd get from a single train/test run with a different seed?
20. Why did you choose bar charts / box plots / confusion-matrix heatmaps specifically, rather than other chart types?

## J. Top 20 Conclusion/Future Scope Questions
1. What's your main conclusion?
2. What's the single most important finding of this project?
3. What future work did you propose, and why those specific directions?
4. What is SMOTE, and how would it help here?
5. What is Optuna, and how would it help here?
6. What is target encoding, and why might it help `last_prod_enq2`/`first_prod_enq2`?
7. Why didn't you implement SMOTE/Optuna/target encoding already, if you knew they'd help?
8. Is 77.9% likely to be a hard ceiling, or could future work meaningfully improve it?
9. What would you need to do to make this project production-ready?
10. What ethical/fairness considerations does this project raise (e.g., bias in `GENDER`/`MARITALSTATUS` features)? *(Not directly addressed in the report — an honest gap if asked)*
11. Would you recommend a bank actually deploy Model B as-is?
12. What monitoring would you add if this were deployed?
13. How would you handle model drift if the bank's applicant population changes over time?
14. What's the very next experiment you'd run if given one more week?
15. Does your future work list address the P3 problem specifically?
16. What's the risk of applying SMOTE to a 4-class problem with an already-hard class (P3)?
17. How would you validate whether Optuna actually breaks the 77-78% plateau, versus your manual search?
18. What's the argument against over-engineering this further, given it's a course project?
19. If this were extended to a full application, what would be the very first non-ML component you'd build?
20. Summarize your conclusion in one sentence.

---

# 14. Rapid-Fire Round (1–3 sentence answers — practice these out loud)

**Q: What is your project?**
A multiclass ML pipeline predicting a bank's 4-tier loan approval priority (P1–P4) by merging internal and external bureau data.

**Q: What problem does it solve?**
Automates risk-tiering of loan applicants from raw behavioral/bureau data instead of manual review.

**Q: What is your objective?**
Build an honest, deployable classifier while identifying and correcting any data leakage in the source features.

**Q: What is your methodology?**
Clean → merge → investigate leakage → engineer features → train/compare 3 classifiers on 2 feature sets → interpret with SHAP → score unseen data.

**Q: What algorithm did you use?**
XGBoost, compared against Logistic Regression and Random Forest baselines.

**Q: Why this algorithm?**
Best accuracy and macro-F1 in both feature settings, consistent with published credit-scoring benchmarks favoring gradient boosting.

**Q: What technologies did you use?**
Pandas, Scikit-learn, XGBoost, SHAP, Matplotlib/Seaborn, Joblib.

**Q: Why these technologies?**
Standard, well-documented, and sufficient for a tabular ML pipeline at this scale — no need for heavier infrastructure.

**Q: What are your modules?**
Not modules in a software-engineering sense — it's a linear notebook pipeline: cleaning → merging → feature engineering → training → evaluation → interpretability → scoring.

**Q: What is your main contribution?**
Discovering that `Credit_Score` leaks the target and training/reporting a comparison model that's genuinely deployable instead of hiding the issue.

**Q: What are your results?**
Model B (deployment-safe) XGBoost: 77.9% accuracy, 0.688 macro-F1.

**Q: What is your best result?**
XGBoost outperforms Logistic Regression and Random Forest in both feature settings.

**Q: What are your limitations?**
No k-fold cross-validation, no feature scaling for Logistic Regression, P3 class is weak (F1=0.34), no formal test suite.

**Q: What is your future scope?**
SMOTE for class balance, Optuna for automated tuning, target encoding for high-cardinality categoricals.

**Q: What would you improve?**
P3 classification performance and statistical confidence via cross-validation.

**Q: What happens if your main component (XGBoost) fails/is unavailable?**
Random Forest is the next-best fallback — only ~1–2 points behind XGBoost in both settings, still a solid ensemble alternative.

**Q: What is P1–P4?**
Bank-assigned risk tiers — P1 lowest risk (auto-approve) to P4 highest risk (likely reject).

**Q: What is data leakage, in one line?**
A feature that encodes the answer through a business process rather than genuinely predicting it causally.

**Q: What is SHAP, in one line?**
A game-theory-based method that fairly attributes each prediction to the features that caused it.

**Q: What is macro-F1, in one line?**
The F1 score averaged equally across all classes, regardless of class size.

---

*End of guide. Re-read Sections 2.6, 2.8, and the Section 13 lists the most — they cover ~80% of what an examiner is likely to actually ask.*
