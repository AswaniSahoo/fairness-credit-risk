# Model Card: Credit Risk Scoring (German Credit)

## Model Details

- **Model type**: XGBoost gradient-boosted tree (T0 track: tuned baseline, no fairness intervention)
- **Framework**: scikit-learn pipeline with XGBoost classifier
- **Hyperparameter search**: 60-trial Optuna search, 5-fold cross-validated ROC-AUC on the training block
- **Selected configuration**: max_depth 5, 350 estimators, learning_rate 0.0055, scale_pos_weight 2.03
- **Threshold**: 0.5 (single global threshold applied to all applicants)
- **Features**: 47 encoded features from 18 input columns (5 numeric, 4 ordinal, 9 nominal one-hot encoded)
- **Version**: Recorded 2026-07-26. Split fingerprint `5f29293fb76300c1`.

## Intended Use

Credit risk scoring for research and education. The model predicts whether a loan applicant is likely to default, outputting a probability that is thresholded at 0.5 into a binary decision.

This is a case study in measuring fairness-performance tradeoffs under realistic constraints, not a deployable credit scoring system.

## Out-of-Scope Use

- Production credit decisions on real applicants.
- Any context where the 200-row test block would be treated as sufficient validation.
- Populations outside the German Credit dataset's demographic (West Germany, 1973-1975).
- Jurisdictions where the ECOA/Regulation B framework does not apply.

## Training Data

- **Source**: UCI Statlog German Credit dataset (Hofmann). 1000 loan applications.
- **Provenance**: `data/raw/german.data`, label-encoded alphabetically per code. Mapping asserted in `tests/integration/test_german_provenance.py`.
- **Split**: 600 train / 200 calibration / 200 test, stratified by target and gender.
- **Class distribution**: 70% good credit (favorable), 30% default.
- **Protected attributes**: Gender (binary: 0 female, 1 male), foreign_worker (binary).

## Evaluation Data

- **Test block**: 200 rows.
- **Group composition**: 138 privileged (male), 62 unprivileged (female).
- **Limitation**: 62 women in the test block. A single flipped prediction changes the female selection rate by 1.6 percentage points.

## Metrics

All values from `reports/track_comparison.json`, T0 (german_credit|T0). Intervals are 95% percentile bootstrap, 2000 replicates stratified within (group, label) cells.

### Performance

| Metric | Point | 95% CI |
|--------|-------|--------|
| ROC-AUC | 0.8296 | [0.7725, 0.8804] |
| Balanced Accuracy | 0.7357 | [0.6679, 0.7988] |
| F1 | 0.6269 | [0.5414, 0.7059] |
| Recall | 0.7000 | [0.5833, 0.8167] |
| Brier Score | 0.1833 | [0.1705, 0.1966] |

Source keys: `runs["german_credit|T0"].intervals.roc_auc`, `.balanced_accuracy`, `.f1`, `.recall`, `.brier`.

### Fairness (Gender: male privileged, female unprivileged)

| Metric | Point | 95% CI |
|--------|-------|--------|
| Disparate Impact | 0.7263 | [0.5342, 0.9106] |
| Statistical Parity Difference | -0.1884 | [-0.3319, -0.0570] |
| Equal Opportunity Difference | -0.1700 | [-0.3401, -0.0150] |
| Equalized Odds Difference | 0.1700 | [0.0621, 0.3800] |

Source keys: `runs["german_credit|T0"].intervals.disparate_impact`, `.statistical_parity_difference`, `.equal_opportunity_difference`, `.equalized_odds_difference`.

### Group Rates

| Group | n | Selection Rate | TPR | FPR |
|-------|---|----------------|-----|-----|
| Male (privileged) | 138 | 0.6884 | 0.8200 | 0.3421 |
| Female (unprivileged) | 62 | 0.5000 | 0.6500 | 0.2273 |

Source keys: `runs["german_credit|T0"].fairness.n_privileged`, `.n_unprivileged`, `.selection_rate_privileged`, `.selection_rate_unprivileged`, `.tpr_privileged`, `.tpr_unprivileged`, `.fpr_privileged`, `.fpr_unprivileged`.

## Fairness Analysis

Four tracks were measured under identical conditions (same split, same encoder, same bootstrap replicates). Only the intervention differs.

| Track | Intervention | Deployable | DI Point | DI 95% CI |
|-------|--------------|------------|----------|-----------|
| T0 | none (control) | yes | 0.7263 | [0.5342, 0.9106] |
| T1 | Kamiran-Calders reweighing | yes | 0.6774 | [0.4848, 0.8793] |
| T2 | ExponentiatedGradient | yes | 0.7263 | [0.5342, 0.9106] |
| T3 | group-specific thresholds | no | 0.6583 | [0.4336, 0.9032] |

Source keys: `runs["german_credit|T0..T3"].intervals.disparate_impact`.

**Findings**:

1. **T1 (reweighing)**: No fairness gain. Disparate impact moved from 0.7263 to 0.6774 (worse). Balanced accuracy dropped from 0.7357 to 0.7143. The reweighing weights (0.858 to 1.080) were too mild to shift the booster's decisions on this split.

2. **T2 (ExponentiatedGradient)**: The constraint did not bind. The base learner's training-block parity gap was -0.0143, already inside the tightest epsilon (0.02). The reduction returned the base learner unchanged. The test-block disparity is a generalisation gap that a training-distribution constraint cannot see.

3. **T3 (group thresholds)**: Marked not deployable. Keying a threshold on the applicant's sex is disparate treatment under ECOA regardless of what it does to a fairness metric.

4. **All intervals overlap**. The disparate-impact interval for every track includes values both above and below 0.8. Compliance with the four-fifths rule cannot be established or refuted from this test block.

## Protected Attribute Handling

Following ECOA (Equal Credit Opportunity Act) and Regulation B:

- **Sex** (`gender`): Prohibited basis. Excluded from the feature matrix. Retained for measurement only.
- **National origin** (`foreign_worker`): Prohibited basis. Excluded from features. Only 37 non-foreign rows in the dataset; intervals on this attribute are near useless.
- **Age**: Permitted in an empirically derived, demonstrably and statistically sound credit scoring system (Regulation B, 12 CFR 1002.6(b)(2)). Retained as a feature.

The encoder enforces this: prohibited-basis columns cannot enter the pipeline by construction (validated in `DatasetSpec.validate()`).

## Operating Point

The served threshold is chosen, not inherited. 0.5 asserts that approving an applicant who
defaults costs the same as declining one who would have repaid; in lending it does not, and at
0.5 the Taiwan baseline catches 36.47 percent of defaults.

- **Selection rule.** The threshold minimising expected misclassification cost at a stated
  false-negative to false-positive ratio.
- **Ratio: 5:1. This is an assumption, not a measurement.** Neither dataset records recovery
  rates or interest margins, so a ratio claiming to be derived from them would be an assumption
  presented as a measurement. It is swept over 1, 2, 5, 10 and 20 in each run record's
  `cost_sensitivity` block so a reader can see how much of the result is the assumption.
- **Fitted on the calibration block, reported on test.** Same discipline as the T3
  post-processing track and for the same reason (finding B4).
- **Effect, German Credit T0**: threshold 0.3606, recall 0.7000 to 0.9667, approval rate 0.34.
  **Taiwan T0**: threshold 0.1899, recall 0.3647 to 0.7272, approval rate 0.6063.
- **It worsens the fairness metrics.** German Credit disparate impact falls from 0.7263 at 0.5
  to 0.5261 at the chosen threshold; Taiwan from 0.9767 to 0.9374. Lowering the decline bar
  moves more applicants into the decline region and the groups are not symmetric around it.
  Reported here because the trade is real and a card that omitted it would be misleading.
- **Sanity check.** At a 1:1 ratio the selector returns 0.5368 on Taiwan and 0.5365 on German
  Credit, which is what threshold 0.5 implicitly assumed all along.

## Serving Contract

- **One global threshold, and it is the chosen one.** The served threshold is the operating
  point above, not 0.5, read from the run record at load time. Responses carry
  `threshold_basis` naming the rule that produced it. There is no group-keyed threshold
  anywhere in the inference path.
- **Prohibited-basis inputs are refused, not ignored.** Supplying `gender`, `foreign_worker` or
  the `personal_status_sex` proxy raises rather than being silently dropped, so a caller cannot
  believe the model considered a factor it must not consider.
- **One inference path.** The API, the Streamlit demo and the tests all call
  `src.serving.predictor.Predictor`; a test asserts the demo and API probabilities agree to
  within 1e-9.
- **No authentication.** Any caller who can reach the service can score an applicant. It must
  not be exposed publicly without an auth layer in front of it.

## Adverse-Action Reason Codes

Regulation B (12 CFR 1002.9) requires a creditor to state the principal reasons for an adverse
action. `POST /predict` with `include_reasons=true` returns up to four reason codes on a
decline, ranked by absolute SHAP contribution. `Predictor.explain` returns them for any
decision and is intended for audit, not for applicant notices.

- **Computed on the encoded matrix**, against the classifier step of the fitted pipeline, so
  the explanation is of the surface the model actually decides on and cannot disagree with the
  prediction it accompanies. Values are for the positive class, which is default.
- **`TreeExplainer` with `model_output="probability"`** for the tree families, against a
  200-row background sample stored inside the track artifact. `KernelExplainer` over a kmeans
  summary for logistic regression.
- **One-hot columns state whether the category applied.** A positive attribution on a category
  the applicant does not hold is normal and meaningful — not holding the strongest
  checking-account status raises the score — so the notice says the category "did not apply"
  rather than naming it as though the applicant held it. Reporting an absent category as
  present would misstate a principal reason.
- **Feature names are encoded column names** such as `status_3`, not applicant-facing prose.
  Mapping them to the sentences a real notice would carry is a product step this repository
  does not take.
- **Approvals carry no reason codes.** ECOA requires them for adverse action; returning them
  on an approval would invite treating them as an eligibility explanation.
- **Cost, measured on German Credit T0**: 0.0231 s per decision with reasons against 0.0043 s
  without, plus roughly 1.1 s once per process to build the explainer on first use.
- **Global feature importance** in each run record is the mean absolute SHAP value over a
  500-row sample of the training block, not the whole block. It is a diagnostic, not a
  published headline metric; no reported performance or fairness number depends on it.

## Track T4: An Externally Scored Model

Google TabFM, a 1.639-billion-parameter tabular foundation model, is reported as track T4 on
German Credit. It is **not trained here and not served here**.

- **How it was produced.** The checkpoint was run once on a GPU and its per-row probabilities
  for this split's test block were recorded, with the checkpoint revision
  (`google/tabfm-1.0.0-pytorch/classification`), commit, backend, device, `n_estimators` and
  seed stored beside them. Everything downstream of the scores is the same code the other
  tracks use.
- **What is verified.** The loader refuses any prediction file whose row identifiers are not
  exactly this split's test block, in order. A file with the right row count, columns and
  value range but shifted identifiers is rejected, and that case is a test.
- **What is taken on trust.** The scores themselves. T4 cannot be re-derived from this
  repository without the checkpoint and a GPU.
- **Result.** ROC-AUC 0.8435 [0.7883, 0.8924] against the tuned booster's 0.8296
  [0.7725, 0.8804], and disparate impact 0.8321 [0.6801, 0.9845] against 0.7263
  [0.5342, 0.9106]. Neither difference is distinguishable on a 200-row test block.
- **Not deployable.** 6.56 GB checkpoint, 0.2662 s per row against 0.000249 s for the booster,
  and the model holds raw training rows at inference time, which is a data-exposure question a
  lender must answer before it reaches production.
- **No operating point.** Selecting one needs calibration-block scores and the offline run
  produced test-block predictions only. T4 is comparable at a matched selection rate but not
  at a cost-selected threshold.

## Second Dataset

The same four tracks were run on the Taiwan credit card default dataset (UCI 350, 30,000 rows),
recorded under `runs["taiwan_credit|T0..T3"]`. It matters for interpreting this card: with 3,622
women and 2,378 men in a 6,000-row test block the intervals are roughly ten times narrower, and
the baseline already sits at disparate impact 0.9767 [0.9594, 0.9934]. There the disadvantaged
group is men, because women default less — which is why privileged and unprivileged values are
per-dataset registry entries rather than global constants.

## Limitations

1. **Sample size**: 200-row test block with 62 women. Statistical power is insufficient to distinguish the tracks from each other or to confirm/deny four-fifths compliance.

2. **Temporal**: Data from West Germany, 1973-1975. Feature distributions, default rates, and gender disparities have no guaranteed relevance to current populations.

3. **Single split**: All results condition on one stratified split (fingerprint `5f29293fb76300c1`). A different seed would produce different point estimates within the reported intervals.

4. **Null results**: None of the three mitigation strategies produced a measurable fairness improvement on this dataset and split. This is an honest outcome, not a failure to try.

5. **T2 score degeneracy**: The ExponentiatedGradient track's mixture concentrated on one predictor (1 nonzero weight out of 3). Its scores take only 2 unique values, so ROC-AUC equals balanced accuracy by construction and cannot be compared to T0's graded ranking.

6. **Post-processing non-deployable**: T3 is reported for comparison only. It cannot be deployed because it constitutes disparate treatment.

## Ethical Considerations

- The model learns from historical lending decisions that may embed past discrimination. A model reproducing those decisions is not "fair" merely because it passes a statistical test.
- The four-fifths rule is a legal threshold, not an ethical one. A disparate impact of 0.81 is not meaningfully different from 0.79.
- Fairness metrics are properties of a model on a population, not guarantees to individuals.
- The 62-woman test subgroup is too small to support confident claims about the model's behaviour on women generally. Reporting rates without group sizes (as this project previously did) obscures that limitation.
