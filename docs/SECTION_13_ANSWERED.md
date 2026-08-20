# Section 13 — Answered
## Companion to VIVA_PREP_GUIDE.md (Top 50 / 25 / 20×7 lists, now with answers)

All answers are 1–3 sentences — viva-speakable, not essay-length. Cross-references to the main guide are given where a topic deserves deeper prep.

---

## A. Top 50 Most Likely Questions — Answered

1. **What is your project about?** Predicting a bank's 4-tier loan approval priority (P1–P4) by merging internal trade-line data with external CIBIL bureau data.
2. **What is `Approved_Flag` / what do P1–P4 mean?** The target label — P1 is lowest risk (auto-approve), P4 is highest risk (likely reject).
3. **What is data leakage and how did you find it?** A feature that encodes the label rather than predicting it causally; found by noticing suspiciously high (99%) accuracy, then confirming `Credit_Score` almost perfectly separates the classes.
4. **Why does `Credit_Score` cause leakage?** Because `Approved_Flag` is essentially a threshold function of `Credit_Score` (P1≈701–809, P2≈669–700, P4≈469–658) — the model would just be re-learning the bank's own cutoff rule.
5. **What are Model A and Model B, and why do both exist?** Model A includes `Credit_Score` (99.2% acc, not deployable); Model B excludes it (77.9% acc, deployable) — both are kept to make the leakage tradeoff explicit rather than hiding it.
6. **What's your final accuracy and macro-F1?** Model B (XGBoost): 77.9% accuracy, 0.688 macro-F1.
7. **Why macro-F1 and not just accuracy?** Classes are imbalanced (P2=60.5%); macro-F1 weights all 4 classes equally, exposing weak minority-class performance accuracy would hide.
8. **Why is the dataset imbalanced, and how much?** P2 = 60.51%, P3 = 15.31%, P4 = 12.51%, P1 = 11.67% — a natural consequence of most applicants being "good but not best" risk.
9. **What algorithm performed best, and why?** XGBoost, in both feature settings — gradient boosting captures non-linear feature interactions better than a linear model, consistent with published credit-scoring benchmarks.
10. **Explain your data cleaning process.** Both datasets use `-99999` instead of `NaN` for missing values; dropped 40 internal rows with the sentinel, dropped 8 external columns that were >70% sentinel, then dropped remaining sentinel rows.
11. **Why drop columns instead of imputing them?** Some columns were missing in over 70% of rows — imputing that much data would fabricate signal, not recover it.
12. **What's the difference between internal and external datasets?** Internal (26 features) is the bank's own trade-line history; external (62 features) is CIBIL bureau data plus the target label.
13. **What is `PROSPECTID`?** The shared applicant ID used to join the two datasets.
14. **Why an inner join?** Only keeps applicants present and clean in both sources, consistent with the independent cleaning already done on each.
15. **What is SHAP and why did you use it?** A game-theory-based method (Shapley values) that attributes each prediction to contributing features; used to verify Model B relies on plausible behavioral signals, not a hidden shortcut.
16. **What are the top SHAP features?** In order: `Age_Oldest_TL`, `enq_L3m`, `max_recent_level_of_deliq`, `num_std_12mts`, `time_since_recent_enq`.
17. **Why is P3 the hardest class?** Its `Credit_Score` range (489–776) overlaps P1, P2, and P4 — genuinely ambiguous even under the bank's own scoring policy.
18. **What does the confusion matrix show for Model B?** Most errors are concentrated between P2/P3 and P3/P4 — adjacent tiers, not distant ones.
19. **What is the "adjacency" property?** Errors land in neighboring risk tiers rather than distant ones, which is a lower business cost mistake for a bank's triage workflow.
20. **Why didn't you scale features for Logistic Regression?** A real gap — not applied; likely contributed to LR's weak 63% accuracy alongside its inherent linearity limitation. See guide §2.7/§5.1.
21. **Why these XGBoost hyperparameters?** Chosen via manual grid evaluation over depth, learning rate, and sampling ratios, guided by cross-validated accuracy — see guide §5.3 for the reasoning behind each value.
22. **Bagging vs boosting?** Bagging (RF) trains trees independently on bootstrapped samples and averages; boosting (XGBoost) trains trees sequentially, each correcting the previous ensemble's errors.
23. **Why a stratified split?** Preserves the same class proportions in train and test, so metrics on a minority class like P1 aren't distorted by an unlucky split.
24. **Why `random_state=42`?** For reproducibility — anyone re-running the notebook gets the identical split and results; 42 has no special meaning beyond being a fixed convention.
25. **What is the unseen dataset, and what did you do with it?** 100 applicants, 42 features, no target, no `Credit_Score` — scored with Model B and saved to `unseen_predictions.csv`.
26. **How did you align its columns with your model?** Applied the same encoding used in training, then used `reindex(columns=DEPLOYMENT_FEATURES, fill_value=0)` to guarantee an exact match.
27. **Why does the unseen dataset lack `Credit_Score`?** It simulates real production scoring — evidence that the intended real-world model shouldn't use that column either.
28. **What if you deployed Model A instead of Model B?** It would fail outright — `Credit_Score` doesn't exist in the unseen data, so Model A can't even generate a prediction for it.
29. **What's your biggest limitation?** No k-fold cross-validation — the 77.9% is a point estimate from a single split, with no confidence interval.
30. **Why didn't you use k-fold cross-validation?** Time/scope constraints for a course project; flagged explicitly as a limitation rather than hidden.
31. **What is `TreeExplainer` and why not `KernelExplainer`?** `TreeExplainer` computes exact SHAP values efficiently for tree models by exploiting tree structure; `KernelExplainer` is slower and model-agnostic, unnecessary here since both models are tree-based.
32. **Why is LR's accuracy so much lower?** It can't capture non-linear feature interactions the ensembles can, compounded by the lack of feature scaling.
33. **What does macro-F1=0.688 tell you that accuracy doesn't?** That performance is uneven across classes — strong on P1/P2/P4 but weak on P3 (F1=0.34), a gap accuracy's P2-dominated average would hide.
34. **Why one-hot for some, ordinal for others?** `EDUCATION` has a natural order (SSC<12TH<...<Post-Graduate) so ordinal preserves it; the rest have no inherent order, so one-hot avoids implying a false ranking.
35. **Why did `PROFESSIONAL`/`OTHERS` get merged into other tiers?** A simplifying judgment call treating them as roughly equivalent risk tiers — not derived from the data, an honest limitation if pressed.
36. **Difference between Model A's and Model B's confusion matrices?** Model A's is close to diagonal (relying on the `Credit_Score` shortcut); Model B's shows realistic adjacent-tier confusion, especially around P3.
37. **How would you improve P3's performance?** Class-balancing (SMOTE) or an engineered feature targeting the ambiguous score band — both listed as future work, not yet implemented.
38. **What future work would you prioritize?** SMOTE for minority classes, Optuna for automated tuning, target encoding for high-cardinality categoricals.
39. **Why 4-class instead of binary?** Reflects the bank's real triage workflow — different tiers map to different downstream actions, not a simple accept/reject.
40. **What does "deployment-safe" mean?** The feature set matches the schema of `unseen_dataset.xlsx` exactly — i.e., every feature the model needs is genuinely available at scoring time.
41. **Could there be a second hidden leakage feature?** Not proven absent, but SHAP's plausible attributions (behavioral, not proxy-like) are indirect evidence against it — an honest limitation, not a guarantee.
42. **Why sample only 500 rows for SHAP?** Computational cost — SHAP scales with dataset size; 500 is representative enough for stable global rankings.
43. **Exact `Credit_Score`-by-class numbers?** P1: 701–809 (mean 716.22); P2: 669–700 (682.40); P3: 489–776 (666.82); P4: 469–658 (645.35).
44. **Why is the RF-vs-XGBoost gap smaller than LR's gap?** Both RF and XGBoost are tree ensembles capable of non-linearity — the ensemble structure itself matters more than the specific boosting mechanism, while LR is fundamentally linear regardless of tuning.
45. **What tools/libraries did you use?** Pandas, Scikit-learn, XGBoost, SHAP, Matplotlib/Seaborn, Joblib, openpyxl — see guide §4 for why each.
46. **What if a new applicant has an unseen category?** The `reindex(fill_value=0)` step silently zero-fills it rather than raising an error — a real robustness gap, not handled explicitly.
47. **How did you save your trained models?** `joblib.dump()` to `model_full_with_credit_score.pkl` and `model_deployment.pkl`.
48. **What testing did you perform?** Stage-by-stage sanity checks (row/column counts after cleaning, SHAP tensor shape, unseen-column alignment) — not a formal automated test suite; see guide §9.
49. **Business cost: P1→P4 vs P2→P3 misclassification?** P1→P4 is a large, costly jump (rejecting a great applicant or approving a terrible one); P2→P3 is a minor tier shift with a much smaller downstream impact — this is exactly the adjacency argument.
50. **First thing you'd add with more time?** K-fold cross-validation, to put a confidence interval around the 77.9% figure.

---

## B. Top 25 Must-Never-Get-Wrong — As Q&A

1. **Q: What do P1–P4 mean?** A: Best (P1) to worst (P4) risk tier.
2. **Q: Why is `Credit_Score` leakage?** A: It's near-deterministic of the target — the model would just re-learn the bank's threshold rule, not real risk.
3. **Q: Which model is deployable, A or B?** A: Model B — Model A depends on a feature absent from the real scoring data.
4. **Q: What's your headline number?** A: 77.9% accuracy, 0.688 macro-F1 (Model B, XGBoost).
5. **Q: Best algorithm overall?** A: XGBoost, in both feature settings.
6. **Q: Why does macro-F1 matter here?** A: It stops the large P2 class from masking poor performance on P1/P3/P4.
7. **Q: Merged dataset size?** A: 42,064 rows × 79 columns.
8. **Q: What does `-99999` represent?** A: The dataset's missing-value sentinel, used instead of `NaN`.
9. **Q: Why is P3 weakest?** A: Its score range overlaps all three other classes.
10. **Q: What is SHAP, conceptually?** A: A fair, game-theory-based way to attribute a prediction to its contributing features.
11. **Q: What's the confusion matrix adjacency finding?** A: Errors cluster between neighboring risk tiers, not distant ones.
12. **Q: Why an inner join on `PROSPECTID`?** A: Keeps only applicants present and clean in both sources.
13. **Q: Why does the unseen dataset exclude `Credit_Score`?** A: It reflects the real production task, confirming that column shouldn't be relied on.
14. **Q: Walk me through the pipeline start to finish.** A: Load → clean `-99999` → merge → EDA/leakage check → encode features → split → train 3 models × 2 feature sets → evaluate → SHAP → score unseen data. (Full detail: guide §6–7.)
15. **Q: Why stratified split?** A: Preserves class ratios in both train and test.
16. **Q: Why does LR underperform?** A: Can't model non-linear interactions; also wasn't feature-scaled.
17. **Q: Bagging vs boosting, one line each?** A: Bagging = independent trees averaged; boosting = sequential trees correcting prior errors.
18. **Q: Was feature scaling applied?** A: No — an acknowledged gap, most relevant to Logistic Regression.
19. **Q: Was cross-validation performed?** A: No — single 80/20 split; acknowledged as a limitation.
20. **Q: What's your tech stack?** A: Pandas, Scikit-learn, XGBoost, SHAP, Joblib.
21. **Q: Class imbalance number?** A: P2 = 60.51% of the data.
22. **Q: What does the 21.3-point gap between Model A and B represent?** A: The accuracy cost of removing the leaky `Credit_Score` feature.
23. **Q: Is this a deployed application?** A: No — a notebook-based analysis pipeline; no architecture/API/database exists.
24. **Q: Top SHAP features?** A: `Age_Oldest_TL`, `enq_L3m`, `max_recent_level_of_deliq`, and others — see guide §2.9.
25. **Q: What does "genuinely deployable" mean here?** A: The feature set matches exactly what's available in the unseen/production data.

---

## C. Top 20 Difficult Questions — Answered

1. **How does SHAP's Shapley-value computation work, conceptually?** It treats each feature as a "player" contributing to a prediction, and computes its average marginal contribution across all possible orderings of features being added — `TreeExplainer` computes this exactly and efficiently for trees rather than via expensive brute-force enumeration.
2. **Why is XGBoost's `max_depth=5` shallower than RF's `max_depth=10`?** In boosting, errors compound across hundreds of sequential trees, so each tree is deliberately kept a "weak learner" to avoid runaway overfitting; RF's independent trees can afford more depth since there's no compounding.
3. **Could a second leakage feature exist?** Not disproven — SHAP's plausible, domain-sensible top features are indirect evidence against it, but this was not formally tested (e.g., via a full leave-one-feature-out audit).
4. **Is the 18% discarded due to the join or the cleaning?** Mostly the cleaning steps (dropping sentinel rows/columns) — the join itself only removes applicants missing from one side, which is a small residual effect.
5. **"Leaky" vs. "very predictive" — what's the difference?** A leaky feature correlates with the label because both come from the same business process (score → policy → label); a genuinely predictive feature has an independent causal relationship to risk that would hold even if the labeling policy changed.
6. **Why does removing `Credit_Score` cost exactly 21.3 points, not more/less?** That's an empirical result specific to this dataset — it reflects how much of the label's variance `Credit_Score` alone explains versus what the remaining 50 behavioral features can recover.
7. **Is there a theoretical ceiling on P3 performance?** Likely yes, bounded by how much the ground-truth label itself overlaps for P3 (its Credit_Score band spans all other classes) — but the exact ceiling wasn't formally estimated (e.g., via Bayes-error approximation).
8. **How would you test whether cleaning introduced bias?** Compare summary statistics (income, age, etc.) of dropped vs. retained rows — if they differ systematically, the cleaning isn't bias-free. Not done in this project; an honest gap.
9. **Why didn't tuning explore `reg_alpha`/`reg_lambda`?** Manual tuning focused on depth/learning-rate/sampling first, as they typically have larger effect; regularization tuning is a reasonable next step, not yet done.
10. **Gini importance vs. SHAP importance — mathematical difference?** Gini importance measures average impurity decrease from splits on a feature (a training-time, model-internal metric with known bias toward high-cardinality features); SHAP measures each feature's actual contribution to individual predictions via Shapley values, giving both local and unbiased global explanations.
11. **How would multicollinearity affect LR specifically?** Correlated features destabilize LR's coefficient estimates (inflated variance, sign flips), while tree models are largely unaffected since they split on one feature at a time regardless of correlation with others.
12. **What does `min_child_weight=3` prevent?** It requires a minimum sum of instance weight in a child node before a further split is allowed, preventing the model from creating overly specific splits on very small, noisy subsets of data.
13. **What would a CV confidence interval likely look like?** Qualitatively, probably a band of a few percentage points around 77.9% (e.g., roughly 76–80%) given the dataset size — not computed exactly in this project.
14. **How does `mlogloss` differ from binary logloss?** `mlogloss` generalizes logloss to multiple classes via a softmax over all classes' predicted probabilities, penalizing the negative log-probability assigned to the true class, instead of a single sigmoid probability for two classes.
15. **Why average SHAP magnitude, not signed SHAP, for global importance?** Signed values would cancel out (positive and negative contributions across samples), hiding a feature's true overall influence; absolute value captures total impact regardless of direction.
16. **Could label noise in `Approved_Flag` itself be limiting the ceiling?** Possible — if some historical approval decisions were inconsistent or manually overridden, that noise would cap any model's achievable accuracy; not tested directly, but plausible given the P3 ambiguity.
17. **Risk of using the same test set for tuning and final evaluation?** It introduces a mild optimistic bias — repeatedly checking test performance during manual tuning means the reported 77.9% may be very slightly inflated versus a strictly held-out final check; a validation-set/CV split would avoid this.
18. **How would the leakage argument change if P3's overlap extended to all classes?** It would actually strengthen the case that `Credit_Score` is a genuine, non-trivial predictor rather than leakage — the overlap pattern (near-total separation except P3) is precisely why it looks like a threshold rule, not organic prediction.
19. **Dimensionality risk from one-hot encoding `last_prod_enq2`/`first_prod_enq2`?** Manageable here since these columns have a modest number of categories; `drop_first=True` also mitigates the increase, though for a truly high-cardinality categorical, target encoding (listed in future work) would scale better.
20. **Would the model need retraining if the bank's policy changes?** Yes — since `Approved_Flag` is itself a product of policy, any change in the bank's approval thresholds would shift the label distribution and require retraining to stay accurate.

---

## D. Top 20 "Why?" Questions — Answered

1. **Why build two models?** To make the leakage tradeoff explicit and defensible instead of silently picking one number.
2. **Why treat `Credit_Score` as leakage?** It near-perfectly separates classes via a threshold pattern matching the bank's own policy, not independent predictive signal.
3. **Why macro-F1 alongside accuracy?** Accuracy is dominated by the majority class (P2); macro-F1 treats all classes equally.
4. **Why XGBoost over Random Forest?** Slightly higher accuracy and macro-F1 in both feature settings, consistent with boosting's error-correction advantage.
5. **Why an inner join?** Keeps only applicants clean and present in both sources.
6. **Why drop sparse columns instead of imputing?** Imputing >70%-missing columns would fabricate signal, not recover it.
7. **Why ordinal `EDUCATION` but one-hot the rest?** `EDUCATION` has genuine order; the others don't.
8. **Why `drop_first=True`?** Avoids the dummy-variable trap / perfect multicollinearity among one-hot columns.
9. **Why `stratify=y`?** Keeps class ratios consistent between train and test.
10. **Why `random_state=42`?** Reproducibility — anyone re-running gets identical results.
11. **Why 80/20 split?** Standard convention giving a large enough test set (~8,400 rows) while retaining most data for training.
12. **Why explain Model B with SHAP, not Model A?** Model B is the deployable model; explaining Model A would just re-confirm `Credit_Score` dominates, which is already known.
13. **Why only 500 SHAP samples?** Computational cost — 500 is representative without excessive runtime.
14. **Why `max_depth` differs between RF (10) and XGBoost (5)?** Boosting compounds errors across sequential trees, so shallower "weak learner" trees are safer; RF's independent trees can afford more depth.
15. **Why no feature scaling for LR?** A genuine gap — not applied, likely part of why LR underperforms.
16. **Why no k-fold cross-validation?** Time/scope constraints for a course project — acknowledged limitation.
17. **Why 4-class instead of binary?** Matches the bank's real triage workflow, not just accept/reject.
18. **Why does the unseen dataset matter methodologically?** It's the evidence confirming Model B's feature set is the intended real-world one.
19. **Why is P3 not "cleanly" solvable?** Its ground-truth score range overlaps all other classes.
20. **Why prefer a 78% deployable model over a 99% one?** The 99% model can't be applied to real scoring data and mostly re-learns the bank's own rule rather than genuine risk patterns.

---

## E. Top 20 "How?" Questions — Answered

1. **How did you discover the leakage?** Suspiciously high baseline accuracy → grouped `Credit_Score` by class → saw near-total separation → confirmed the unseen dataset excludes it.
2. **How does XGBoost's gradient boosting work?** Sequentially adds trees, each fitting the gradient of the loss with respect to current predictions (i.e., correcting residual errors), with regularization to control overfitting.
3. **How does Random Forest differ from XGBoost mechanically?** RF trains trees independently on bootstrapped samples and averages (bagging); XGBoost trains trees sequentially, each correcting prior errors (boosting).
4. **How does SHAP's TreeExplainer compute attributions efficiently?** Exploits the known structure of decision trees to compute exact Shapley values via a polynomial-time algorithm, avoiding the exponential brute-force cost Shapley values would otherwise require.
5. **How did you clean the `-99999` values?** Row-drop for the internal dataset's 40 affected rows; column-drop (>10,000 missing) plus row-drop for the remainder in the external dataset.
6. **How did you merge the datasets?** `pd.merge(df_internal, df_external, how='inner', on='PROSPECTID')`.
7. **How did you encode categorical features?** Ordinal mapping for `EDUCATION`; one-hot (`drop_first=True`) for the rest; label encoding for the target.
8. **How did you evaluate across 6 model×feature-set combinations?** A reusable `train_and_evaluate()` function computing accuracy, macro-F1, classification report, and confusion matrix for each.
9. **How did you align the unseen dataset's schema?** `reindex(columns=DEPLOYMENT_FEATURES, fill_value=0)` after applying identical encoding.
10. **How did you save your trained models?** `joblib.dump()` to two separate `.pkl` files.
11. **How does the confusion matrix reveal adjacency?** Off-diagonal errors are concentrated near the diagonal (P2↔P3, P3↔P4) rather than in far corners (P1↔P4).
12. **How would you explain one applicant's prediction using SHAP?** Pull that row's SHAP values from the tensor and show which features pushed the prediction toward their assigned class, and by how much.
13. **How does class imbalance affect accuracy vs. macro-F1 differently?** Accuracy is pulled toward the majority class's performance; macro-F1 weights every class equally regardless of size.
14. **How did you decide the 10,000-row column-drop threshold?** A pragmatic heuristic from inspecting the missingness distribution, not a derived formula.
15. **How does Logistic Regression handle 4 classes?** Via a multinomial/softmax extension that models each class's log-odds relative to the others.
16. **How would you add SMOTE to this pipeline?** Apply it to the training split only (after the train/test split, to avoid leakage into test data), oversampling minority classes (P1, P3, P4) before fitting the classifier.
17. **How would cross-validation change your evaluation?** It would replace the single 80/20 split with multiple folds, giving a mean and variance for accuracy/macro-F1 instead of one point estimate.
18. **How does one-hot encoding avoid false ordinal relationships?** Each category gets its own binary column, so no numeric ordering is imposed between unordered categories like `GENDER` or `MARITALSTATUS`.
19. **How did you verify the unseen-scoring step was correct?** Confirmed zero unaccounted columns between `unseen_dataset.xlsx`'s base columns and the engineered feature set before scoring.
20. **How would you explain this project to a non-technical bank manager?** "We built a model that sorts loan applicants into 4 risk groups using their credit history, and made sure it doesn't just copy the bank's existing scoring rule — it actually learns from real applicant behavior."

---

## F. Top 20 "What If?" Questions — Answered

1. **What if `Credit_Score` had less class overlap?** The leakage claim would be even stronger — near-total separation is already close to a pure threshold rule.
2. **What if the dataset were 10x larger?** Might reconsider LightGBM for training speed; core methodology (cleaning, leakage-check, dual-model comparison) would stay the same.
3. **What if a new applicant has an unseen category?** It gets silently zero-filled by `reindex`, rather than raising an error — a real robustness gap.
4. **What if you'd used a left join instead?** Internal-only records without a matching external row would get nulls, reintroducing missing values the cleaning step had just resolved.
5. **What if you'd imputed the 8 dropped columns instead?** You'd invent the majority of each column's values, likely adding noise rather than genuine signal.
6. **What if class imbalance were more extreme (P2=90%)?** Macro-F1 would become even more essential, and minority-class recall would likely worsen further without techniques like SMOTE.
7. **What if XGBoost weren't available?** Random Forest is the next-best choice — only 1–2 points behind in both settings.
8. **What if the bank's approval policy changes after deployment?** The model would need retraining, since `Approved_Flag` is itself a product of that policy.
9. **What if you removed SHAP entirely?** You'd lose the main evidence that Model B relies on plausible behavioral drivers rather than a hidden shortcut.
10. **What if you only had the internal dataset?** You'd lose the target label entirely (`Approved_Flag` lives in the external dataset) — the project wouldn't be possible without the merge.
11. **What if `Credit_Score` were included in the unseen dataset after all?** You'd use Model A instead — but this wasn't the case here, which is exactly why Model B was built.
12. **What if your test-set metrics don't generalize?** That's the core risk of not having cross-validation — a single split's result could be somewhat optimistic or pessimistic by chance.
13. **What if P3 could never exceed 40% F1 no matter the model?** That would suggest the ceiling is in the label itself (inherent ambiguity), not a modelling deficiency — consistent with what was already observed.
14. **What if asked to explain with zero jargon?** "We taught a computer to guess how risky a loan applicant is, using their past credit behavior — and made sure it wasn't just copying a shortcut answer instead of really learning."
15. **What if real-time scoring were required instead of batch?** The `.pkl` model could be wrapped in a lightweight API, but no such deployment layer exists currently.
16. **What if a stakeholder demanded only the 99% model be reported?** You'd need to explain clearly that it can't be applied to real applicants (missing feature) and mostly restates existing policy — recommend Model B regardless of pressure.
17. **What if your one-hot columns had 100+ categories?** Dimensionality would blow up; target encoding (already listed as future work) would be the better choice at that scale.
18. **What if you had to explain a rejection to an individual applicant?** Pull their SHAP values to show which specific factors (e.g., recent enquiries, delinquency level) drove their classification.
19. **What if you found a cleaning-logic bug after submission?** Acknowledge it directly, explain the fix, and note whether it would materially change the headline numbers — honesty here matters more than the bug itself.
20. **What if asked to justify every hyperparameter?** Be ready with the reasoning in guide §5.3 for each XGBoost parameter — depth, learning rate, subsample, colsample, min_child_weight.

---

## G. Top 20 Technology Questions — Answered

1. **What is Pandas, and why?** Tabular data library; used for all cleaning/merging/feature engineering at ~50k-row scale.
2. **What is XGBoost, and why over alternatives?** Gradient-boosted tree library; chosen for best empirical accuracy, and it's the most widely documented for tabular ML.
3. **What is Scikit-learn used for?** Train/test splitting, encoding, Logistic Regression, Random Forest, and evaluation metrics.
4. **What is SHAP, and why over LIME?** Game-theory-based model interpretability; chosen over LIME because Shapley values give theoretically consistent, exact (for trees) attributions rather than local approximations.
5. **Why Matplotlib/Seaborn over Plotly?** Static plots are sufficient for a PDF report; no interactivity was needed.
6. **What is Joblib, and why not raw Pickle?** Model serialization; more efficient than Pickle for objects containing large NumPy arrays like fitted trees.
7. **What is `openpyxl`, and why needed?** The engine Pandas uses to read `.xlsx` files — required since the source data is Excel format.
8. **Why Jupyter Notebook as the environment?** Enables interleaving code, results, and explanation in one document — ideal for an analysis-and-report workflow.
9. **What Python/library versions were used?** Pinned in `requirements.txt` (e.g., XGBoost, SHAP, scikit-learn versions) for reproducibility.
10. **Why not a neural network?** Tabular data with ~50 features and 42k rows is exactly where gradient-boosted trees typically outperform deep learning, which needs much more data to justify its complexity.
11. **Why not CatBoost, given native categorical support?** Not tested — XGBoost was sufficient; CatBoost's native categorical handling is a reasonable future comparison.
12. **Why not LightGBM for speed?** Dataset size (~42k rows) isn't large enough for LightGBM's histogram-based speed advantage to matter much.
13. **What does `verbosity=0` do?** Suppresses XGBoost's training log output for a cleaner notebook.
14. **What does `eval_metric='mlogloss'` mean?** Multiclass log-loss — the loss function XGBoost optimizes internally for this 4-class problem.
15. **Why `n_jobs=-1` in Random Forest?** Uses all available CPU cores to parallelize tree training, speeding up fitting.
16. **`LabelEncoder` vs `OneHotEncoder` — role of each here?** `LabelEncoder` was used for the ordinal `EDUCATION` mapping and the target; one-hot (`pd.get_dummies`) for unordered categoricals.
17. **Why `output_dict=True` in `classification_report`?** Allows programmatic extraction of macro-F1 for comparison across the 6 model runs, instead of just printing text.
18. **What does `ConfusionMatrixDisplay.from_predictions` do?** Directly plots a confusion matrix from true/predicted labels without manually computing and formatting the matrix.
19. **Why no `ColumnTransformer`/pipeline object?** Manual encoding was simpler to inspect and debug for this project's scale; a pipeline would be a good refactor for production use.
20. **What would change with PyTorch/TensorFlow instead?** Likely worse performance-to-effort ratio on this tabular dataset size, plus far more implementation complexity for equivalent or lower accuracy.

---

## H. Top 20 Results/Evaluation Questions — Answered

1. **Final accuracy and macro-F1?** 77.9% accuracy, 0.688 macro-F1 (Model B, XGBoost).
2. **Accuracy gap between Model A and B, and why?** 21.3 points — entirely attributable to removing `Credit_Score`.
3. **Model B's per-class precision/recall/F1?** P1: 0.83/0.78/0.80; P2: 0.82/0.92/0.87; P3: 0.43/0.28/0.34; P4: 0.76/0.71/0.74.
4. **Lowest-recall class, and why?** P3 (recall 0.28) — its score range overlaps all other classes.
5. **Why is P2's F1 so much higher than P3's?** P2 has far more training examples (5,090 in test) and a cleanly separated score range, unlike P3.
6. **What does "support" mean?** The number of true instances of that class in the test set (e.g., P2 support = 5,090).
7. **Total test samples?** 8,413 (20% of 42,064).
8. **Naive baseline accuracy, and comparison?** ~60.5% (always predict P2); Model B at 77.9% clearly beats this baseline.
9. **Is 77.9% good for this problem type?** Yes — consistent with published benchmarks (75–80%) for structurally similar 4-class bureau-based credit datasets.
10. **How consistent were results across XGBoost configs?** Accuracy plateaued in a narrow 77–78% band across several manually tested configurations.
11. **Distance from a "perfect" confusion matrix?** Far from perfect on P3 specifically; near-diagonal (good) for P1, P2, P4.
12. **Why is the RF/XGBoost gap larger in Model B than A?** With `Credit_Score` present (Model A), both models can lean heavily on one dominant feature, converging to similar high performance; without it (Model B), the models must rely on more nuanced feature interactions, where boosting's error-correction gives XGBoost a slightly larger edge.
13. **Real-world cost interpretation of adjacency?** Misclassifying into a neighboring tier is a cheaper mistake than a distant one, since neighboring tiers likely trigger similar downstream review actions.
14. **How would 5-fold CV change results?** Would give a mean and spread for accuracy/macro-F1 instead of one point estimate — not done, an acknowledged limitation.
15. **Metric priority if false negatives were costlier?** Recall (especially for P4, the highest-risk class) would be prioritized over precision.
16. **Why no ROC-AUC reported?** Not included in this report's evaluation; would need a one-vs-rest extension for the multiclass setting — a reasonable addition for future work.
17. **How do you know it's not overfit to this split?** You don't with full certainty — this is exactly why the lack of cross-validation is flagged as a limitation.
18. **What does macro-F1=0.688 mean in plain English?** On average across all 4 risk tiers treated equally, the model gets a "balanced correctness" score of about 69%, pulled down mainly by weak P3 performance.
19. **Would 77.9% likely hold with a different seed?** Probably close, within a point or two, but this wasn't formally verified via multiple seeds/CV.
20. **Most important number in the results section?** The 77.9%/0.688 pair for Model B — everything else (Model A's 99%, per-class breakdown) exists to contextualize or explain it.

---

## I. Top 20 Figures/Tables/Diagrams Questions — Answered

1. **Table I — what and why?** Target class distribution after cleaning; establishes the imbalance that motivates macro-F1.
2. **Table II — how does it support the leakage claim?** Shows `Credit_Score` ranges per class are almost non-overlapping except for P3 — direct numerical evidence of near-determinism.
3. **Fig 2 (boxplot) — what does P3 overlap show?** Visually, P3's box/whiskers span across P1, P2, and P4's ranges, unlike the other three classes' tight separation.
4. **Table III — why compare 6 combinations?** To isolate the leakage effect cleanly by comparing the *same* three algorithms across *both* feature sets, not just reporting one best number.
5. **Fig 3 — visual takeaway?** Ensembles (RF, XGBoost) clearly outperform Logistic Regression in both settings, and the accuracy gap between Model A and B is visually obvious.
6. **Table IV — why break down by class?** A single blended accuracy hides that P3 is much weaker than the other three classes — this table exposes that.
7. **Fig 4 — difference between the two matrices?** Model A's is near-diagonal (relies on the `Credit_Score` shortcut); Model B's shows realistic, adjacency-clustered confusion.
8. **What does "adjacency" mean concretely in Fig 4?** Off-diagonal error mass sits close to the diagonal (e.g., P2 predicted as P3) rather than in the far corners (e.g., P1 predicted as P4).
9. **Fig 5 — what does bar length mean?** Mean absolute SHAP value for that feature, averaged across all 4 classes and all sampled applicants — a measure of overall influence on predictions.
10. **Why does Fig 5 show ~15 features, not all 50?** Kept to the most influential subset for a readable chart — the remaining ~35 features have comparatively negligible SHAP contribution.
11. **Is Fig 1 redundant with Table I?** No — Fig 1 gives an immediate visual sense of the imbalance's scale, while Table I gives the precise numbers; they serve complementary purposes.
12. **Why no system architecture diagram?** The project is a linear notebook pipeline, not a multi-component system — the section headers and Section III/IV text already describe the flow; a diagram would be somewhat redundant given the page-length constraint. (A verbal ASCII version is in the main guide §7 if needed live.)
13. **Is there a P4-specific SHAP plot available?** Yes — generated in the notebook (`plots/07_shap_p4_class.png`) but not included in the final 4-page report due to space; can be shown if asked.
14. **Why no ROC curve?** Not included in this report's evaluation scope; a reasonable future addition, especially with a one-vs-rest multiclass extension.
15. **Why no learning curve (train vs. test error over iterations)?** Not generated — would be a useful diagnostic for detecting over/underfitting during training, listed as a possible improvement if asked.
16. **One more figure you'd add?** A P3-focused SHAP or feature-distribution plot specifically isolating what makes P3 ambiguous, since it's the project's main unresolved weak point.
17. **What conclusion can you draw from Fig 4 alone?** That Model B's mistakes are systematic and tier-adjacent rather than random — supporting its practical deployability despite a lower raw accuracy than Model A.
18. **Would Table II's numbers change with a different random seed?** No — Table II summarizes the full cleaned dataset (not a train/test split), so it's independent of the modelling random seed.
19. **Difference between Table III's numbers and a different-seed rerun?** Table III's exact figures come from one fixed split (`random_state=42`); a different seed would likely shift each number by a small amount, which is exactly why cross-validation would be a valuable addition.
20. **Why these specific chart types (bar/box/heatmap)?** Bar charts for categorical comparisons (class counts, model accuracy), box plots for showing a numeric distribution's spread by group (`Credit_Score` by class), and confusion-matrix heatmaps as the standard way to visualize multiclass prediction errors.

---

## J. Top 20 Conclusion/Future Scope Questions — Answered

1. **Main conclusion?** A deployment-safe XGBoost model achieves 77.9% accuracy/0.688 macro-F1 after correctly identifying and removing a leakage feature.
2. **Single most important finding?** That `Credit_Score` leaks the target, and that this must be corrected before reporting any headline accuracy number.
3. **Future work proposed, and why?** SMOTE (class balance), Optuna (automated tuning), target encoding (high-cardinality categoricals) — each targets a specific, already-diagnosed weakness in the current pipeline.
4. **What is SMOTE?** Synthetic Minority Oversampling Technique — generates synthetic examples of minority classes (P1, P3, P4) rather than simply duplicating existing ones, to rebalance training data.
5. **What is Optuna?** An automated hyperparameter optimization framework using smarter-than-grid-search methods (e.g., Bayesian/TPE sampling) to more efficiently search the hyperparameter space.
6. **What is target encoding, and why might it help?** Replaces a category with the mean target value for that category; could better capture signal from `last_prod_enq2`/`first_prod_enq2` than one-hot encoding if those have higher cardinality than initially assumed.
7. **Why weren't these already implemented?** Time/scope constraints — priority was given to correctly diagnosing and handling the leakage issue first, since that affects the validity of any result reported.
8. **Is 77.9% a hard ceiling, or could future work improve it?** Likely close to a ceiling given manual tuning already plateaued in a narrow band, but formal automated search (Optuna) hasn't been exhaustively tried, so it's not proven to be a hard limit.
9. **What's needed to make this production-ready?** Cross-validation for confidence, input validation for unseen categories, a real-time scoring wrapper/API, and monitoring for data/label drift.
10. **What ethical/fairness considerations does this raise?** Not directly addressed in the report — features like `GENDER` and `MARITALSTATUS` are used, and their fairness impact on predictions wasn't audited; an honest gap if asked.
11. **Would you recommend deploying Model B as-is?** With caveats — it's the honest, schema-correct choice, but should first get cross-validated confidence intervals and a fairness audit before real production use.
12. **What monitoring would you add if deployed?** Tracking prediction distribution drift over time and periodically re-validating against new labeled outcomes, to catch if the bank's policy or applicant pool shifts.
13. **How would you handle model drift?** Scheduled retraining using fresh labeled data, plus monitoring for feature-distribution shifts (e.g., in `enq_L3m` or `Age_Oldest_TL`) that would signal the model is going stale.
14. **Next experiment with one more week?** Run automated cross-validation to get a confidence interval around 77.9%, since that most directly strengthens the report's core claim.
15. **Does future work address P3 specifically?** Indirectly — SMOTE targets minority-class imbalance broadly (P1, P3, P4), though a P3-specific investigation (e.g., isolating what separates it from P2) isn't explicitly proposed.
16. **Risk of applying SMOTE to an already-ambiguous class like P3?** SMOTE could generate synthetic P3 examples that reinforce the existing overlap with P1/P2/P4 rather than resolving it, since the ambiguity is in the label itself, not just class size.
17. **How would you validate Optuna beats manual search?** Compare cross-validated accuracy/macro-F1 of the Optuna-tuned model against the current manually-tuned XGBoost on the same held-out test set.
18. **Argument against over-engineering further, given it's a course project?** The current scope already demonstrates the core skill being assessed — leakage detection and honest evaluation — additional engineering has diminishing returns for a course deliverable versus a production system.
19. **First non-ML component if extended to a full application?** A lightweight scoring API wrapping the saved `.pkl` model, so it could actually be called by another system instead of only running inside a notebook.
20. **One-sentence conclusion?** A deployment-safe XGBoost model, built after correctly diagnosing and removing a data leakage feature, predicts bank loan approval priority with 77.9% accuracy and 0.688 macro-F1, with SHAP confirming it relies on genuine behavioral risk signals.

---

*This file completes the answers for every "questions-only" list in Section 13 of VIVA_PREP_GUIDE.md. Read alongside the main guide — deeper explanations for the trickier items here are cross-referenced back to their full treatment in Sections 2–9.*
