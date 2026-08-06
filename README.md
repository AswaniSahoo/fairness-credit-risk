# Fairness-Aware Credit Risk Scoring

Four fairness interventions and a tabular foundation model, measured against a tuned baseline
under identical conditions, on two datasets. The interesting result is that none of them helped
much, and the reasons why are different for each. The decision threshold is chosen by expected
cost rather than left at 0.5, which turns out to buy a great deal of recall and cost fairness.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-235%20passing-green.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Every number in this README is generated from [`reports/track_comparison.json`](reports/track_comparison.json)
by [`scripts/generate_report_assets.py`](scripts/generate_report_assets.py). None is typed by
hand. An earlier version of this project published a comparison chart whose values appear in no
artifact at all, which is why that rule now exists and is enforced by a test.

---

## What this measures

A credit risk model can be unfair in ways that a single accuracy number hides. The standard
remedies act at three different stages, and the literature reports them working. This project
implements five tracks that differ **only** in the intervention, and measures what each one
actually buys.

| Track | Intervention | Stage | Deployable |
|---|---|---|---|
| T0 | none | control | yes |
| T1 | Kamiran-Calders reweighing, recomputed per fold | pre-processing | yes |
| T2 | fairlearn `ExponentiatedGradient`, demographic parity | in-processing | yes, with caveats |
| T3 | group-specific decision thresholds | post-processing | **no** |
| T4 | Google TabFM, a tabular foundation model, scored offline | external model | **no** |

T4 is not trained here and not served here. A 6.56 GB checkpoint was run once on a GPU and its
per-row probabilities for this split's test block were recorded, along with the checkpoint
revision, backend, device and seed. Everything downstream of those scores is the same code
every other track uses, and the loader refuses any prediction file whose row identifiers are
not exactly this split's test block. That check is what separates an external result from a
number typed into an artifact.

T3 is measured and reported but marked not deployable: keying a credit decision on the
applicant's sex is disparate treatment under ECOA and Regulation B whatever it does to a
fairness metric. It is included because this project originally shipped exactly that rule
behind its API, and documenting why it was removed is more useful than deleting it silently.

Comparability is structural, not aspirational. All tracks share one seeded, fingerprinted
split artifact, one encoder, one metric implementation, and one set of bootstrap replicates.

---

## Results

### German Credit (1,000 rows, 200-row test block, 62 women in it)

<!-- generated from reports/tables/track_comparison_german_credit.md -->

| Track | Intervention | Stage | Deployable | ROC-AUC | Balanced Accuracy | Disparate Impact | Stat. Parity Diff. |
|---|---|---|---|---|---|---|---|
| T0 | none | control | yes | 0.8296 [0.7725, 0.8804] | 0.7357 [0.6679, 0.7988] | 0.7263 [0.5342, 0.9106] | -0.1884 [-0.3319, -0.0570] |
| T1 | Kamiran-Calders reweighing | pre-processing | yes | 0.8298 [0.7733, 0.8819] | 0.7143 [0.6464, 0.7774] | 0.6774 [0.4848, 0.8793] | -0.2151 [-0.3537, -0.0755] |
| T2 | ExponentiatedGradient, demographic parity | in-processing | yes | 0.7357 [0.6679, 0.7988] [*] | 0.7357 [0.6679, 0.7988] | 0.7263 [0.5342, 0.9106] | -0.1884 [-0.3319, -0.0570] |
| T3 | group-specific thresholds | post-processing | no | 0.8296 [0.7725, 0.8804] | 0.7333 [0.6738, 0.7881] | 0.6583 [0.4336, 0.9032] | -0.1758 [-0.3055, -0.0483] |
| T4 | tabular foundation model, scored offline | external model | no | 0.8435 [0.7883, 0.8924] | 0.7155 [0.6476, 0.7833] | 0.8321 [0.6801, 0.9845] | -0.1302 [-0.2576, -0.0117] |

[*] T2 emits a decision, not a graded score. Its ROC-AUC equals its balanced accuracy by
construction, so it is not comparable with the other tracks' ranking quality.

TabFM posts the best ROC-AUC and the best disparate impact, and neither is distinguishable
from the tuned baseline: 0.8435 against 0.8296 with intervals that overlap heavily, and 0.8321
against 0.7263 with intervals that both span 0.8. On 200 test rows a 1.4-point AUC difference
is noise. The honest summary is that a 1.6-billion-parameter foundation model, costing 0.2662 s
per row against 0.000249 s, did not distinguishably beat a tuned gradient booster on this
dataset.

![Fairness-accuracy tradeoff, German Credit](reports/figures/tradeoff_german_credit.png)

**No intervention improved fairness.** T1 made disparate impact worse (0.6774 against 0.7263)
and cost balanced accuracy. T3 also made it worse. T2 left the predictions untouched. Every
interval spans the 0.8 four-fifths line, so on this dataset none of these differences is
distinguishable from noise in the first place.

The per-group rates show where the disparity sits, with the group sizes on the bars because a
rate on 62 people and a rate on 138 are not the same kind of number:

![Per-group selection, TPR and FPR by track, German Credit](reports/figures/group_rates_german_credit.png)

And the same results as intervals rather than points, which is the honest way to read them:

![Disparate impact intervals by track, German Credit](reports/figures/intervals_german_credit.png)

Every bar crosses the 0.8 line. A reader given only the point estimates would conclude the
baseline fails the four-fifths rule at 0.7263 and that reweighing makes it worse; the intervals
say the data cannot support either statement.

### Taiwan credit default (30,000 rows, 6,000-row test block, 3,622 women and 2,378 men in it)

<!-- generated from reports/tables/track_comparison_taiwan_credit.md -->

| Track | Intervention | Stage | Deployable | ROC-AUC | Balanced Accuracy | Disparate Impact | Stat. Parity Diff. |
|---|---|---|---|---|---|---|---|
| T0 | none | control | yes | 0.7902 [0.7755, 0.8037] | 0.6629 [0.6500, 0.6756] | 0.9767 [0.9594, 0.9934] | -0.0209 [-0.0366, -0.0060] |
| T1 | Kamiran-Calders reweighing | pre-processing | yes | 0.7898 [0.7754, 0.8032] | 0.6625 [0.6494, 0.6756] | 0.9779 [0.9615, 0.9940] | -0.0198 [-0.0348, -0.0054] |
| T2 | ExponentiatedGradient, demographic parity | in-processing | yes | 0.6589 [0.6456, 0.6712] [*] | 0.6589 [0.6456, 0.6712] | 0.9800 [0.9636, 0.9961] | -0.0180 [-0.0327, -0.0035] |
| T3 | group-specific thresholds | post-processing | no | 0.7902 [0.7755, 0.8037] | 0.7220 [0.7082, 0.7352] | 0.9425 [0.9137, 0.9712] | -0.0426 [-0.0649, -0.0210] |

![Fairness-accuracy tradeoff, Taiwan](reports/figures/tradeoff_taiwan_credit.png)

Here the intervals are roughly ten times narrower, and the picture is clear rather than
ambiguous: **the baseline already sits at disparate impact 0.9767**, so there is almost nothing
to mitigate. T1 and T2 move it by less than 0.004. T3 makes it worse while improving balanced
accuracy, because its real effect is to re-pick the operating point rather than to equalise
anything.

The group rates make T3's mechanism visible. It raises the true positive rate by moving both
groups' cut-offs, not by closing the gap between them:

![Per-group selection, TPR and FPR by track, Taiwan](reports/figures/group_rates_taiwan_credit.png)

And the intervals, on the same scale as the German Credit plot above, are what statistical power
actually looks like:

![Disparate impact intervals by track, Taiwan](reports/figures/intervals_taiwan_credit.png)

No bar comes near 0.8. Compare this with the German Credit forest plot: same code, same metric,
same number of bootstrap replicates, thirty times the test block.

---

## The finding worth taking away

Put the two datasets side by side:

- German Credit has visible disparity (0.7263) and **62 women in its test block**. One flipped
  prediction moves the female approval rate by 1.6 percentage points. Nothing can be
  established there, in either direction.
- Taiwan has 3,622 women and 2,378 men in its test block, enough power to resolve differences
  of 0.02, and its baseline disparity is already inside the legal threshold with room to spare.
  Note that the disadvantaged group there is **men**: women default less, so the ratio runs the
  other way, which is why the privileged value lives in a per-dataset registry.

The dataset where fairness is measurable turns out not to need fixing, and the dataset that
looks unfair cannot support any conclusion about whether it is. Most published fairness results
on German Credit are three-decimal point estimates on a sample this size. That is the problem
this repository is really about.

---

## Dataset bias is not model bias

[`notebooks/01_eda_bias_audit.ipynb`](notebooks/01_eda_bias_audit.ipynb) measures the German
Credit data itself, with no model involved. The recorded outcomes are favorable for 0.7232 of
men and 0.6484 of women, a disparate impact of **0.8966** — which passes the four-fifths rule.
The tuned model's disparate impact on the same attribute is **0.7263**, which does not.

Those are two different quantities, and the project's earlier code confused them: it passed
ground-truth labels to AIF360 and published the result as a model metric. That is why its
reported figure, 0.8903, sits next to the dataset-level 0.8966 rather than next to the model's
0.7263. The notebook now states in its first cell that everything in it is a property of the
data, and the model numbers come only from the comparison artifact.

The notebook also reports the intersectional cells, where the smallest is 23 rows. It prints
those counts next to the rates, because a favorable rate of 0.8261 on 23 people is not a
finding.

---

## Three defects worth reading the code for

Each was found by auditing this project's own earlier version, and each is now pinned by a
regression test that fails against the old behaviour.

**The fairness metrics never measured the model.** The previous implementation built an AIF360
dataset with `label_name='true_label'` and put predictions in a separate column that AIF360
never read, so disparate impact was computed from the ground-truth labels. Proof: feeding the
labels in as predictions reproduces the previously published figures of 0.8903 and −0.0795 to
four decimals. See [`src/evaluation/group_fairness.py`](src/evaluation/group_fairness.py) and
`test_labels_as_predictions_recovers_the_dataset_bias_not_a_model_metric`.

**Model selection maximised the inverse of accuracy.** The Optuna objective read
`predict_proba(...)[:, 0]`, the probability of *good* credit, and scored it against a target
where 1 means default — returning `1 − AUC`. The search ranked models by which was worst for
50 trials. See [`src/training/search.py`](src/training/search.py), where the positive class is
named once and asserted against the fitted estimator's `classes_`.

**Dropping `gender` did not remove sex from the model.** `personal_status_sex` code 1 is A92,
the only female code present in German Credit, so it holds for exactly the 310 female rows and
no male row. The regression test sweeps all 47 encoded columns and fails if any value of any
column matches the female mask exactly.

A fourth is subtler and only showed up while building T2. A demographic parity constraint is
enforced on the data it is measured on, which is the training block. The tuned booster fits
those rows closely enough that its **training-block parity gap is −0.0143, already inside the
tightest constraint**, while its test-block gap is −0.1884. The reduction therefore had nothing
to correct and returned the base learner unchanged at every constraint strength tried. The
disparity is a generalisation gap, and an in-processing constraint cannot see it.

---

## Design decisions

**Protected attributes are excluded from the feature matrix, except age.** Sex and national
origin are prohibited bases under ECOA and Regulation B. Age is different: Regulation B permits
it in an empirically derived, demonstrably and statistically sound credit scoring system, so
`age` is a feature and is still reported on. The distinction lives in data, in
[`src/data/registry.py`](src/data/registry.py), not in prose, and the registry refuses to
validate a specification that lists a prohibited basis as a feature.

**One global threshold at inference.** No group-keyed thresholds; the serving layer refuses a
prohibited-basis field as input rather than silently dropping it, so the contract stays legible.

**The disparity direction is per-dataset.** German Credit encodes sex 0 female / 1 male with
women approved less often; Taiwan encodes it 1 male / 2 female with men defaulting more. A
global `PRIVILEGED_VALUE` constant is wrong for one of the two whatever value it holds.

**Three-way split.** Models fit on train, post-processing fits on calibration, everything
reported comes from test. The original code fitted its group thresholds on the training block's
own probabilities, where a tree ensemble is close to memorising its training set.

**Every metric carries a bootstrap interval**, resampled within (group, label) cells so a
replicate cannot empty a group and make a rate undefined. Tracks share the same replicates, so
comparisons are paired.

---

## Nobody chose 0.5

A probability model does not make decisions. A threshold does. Every result above is reported
at 0.5, and 0.5 is not a neutral default: it asserts that approving an applicant who defaults
costs the same as declining one who would have repaid. In lending it does not. At 0.5 the
Taiwan baseline catches **36.47 percent of defaults**, which is not an operating point anyone
would sign off.

So the threshold is now chosen, by minimising expected misclassification cost at a stated
false-negative to false-positive ratio, fitted on the **calibration** block and reported on
test. Same discipline as T3's post-processing, and for the same reason (finding B4): a
threshold picked on the block it is scored on has seen the answer.

<!-- generated from reports/tables/operating_point_taiwan_credit.md -->

**Taiwan**

| Track | Threshold | Approval rate | Recall at 0.5 | Recall at threshold | Disparate impact |
|---|---|---|---|---|---|
| T0 | 0.1899 | 0.6063 | 0.3647 | 0.7272 | 0.9374 |
| T1 | 0.1879 | 0.5995 | 0.3640 | 0.7340 | 0.9414 |

<!-- generated from reports/tables/operating_point_german_credit.md -->

**German Credit**

| Track | Threshold | Approval rate | Recall at 0.5 | Recall at threshold | Disparate impact |
|---|---|---|---|---|---|
| T0 | 0.3606 | 0.3400 | 0.7000 | 0.9667 | 0.5261 |
| T1 | 0.3874 | 0.3950 | 0.7000 | 0.9500 | 0.7048 |

Default capture roughly doubles on Taiwan, 0.3647 to 0.7272, and the approval rate falls from
0.8890 to 0.6063. That is the trade, stated rather than hidden.

**And it makes fairness worse.** German Credit's disparate impact drops from 0.7263 at 0.5 to
**0.5261** at the chosen threshold; Taiwan's falls from 0.9767 to 0.9374. Lowering the bar for
declining pushes more applicants into the decline region, and the groups do not sit
symmetrically around it. A project optimising for a good-looking fairness number would report
the 0.5 figure and stop. The two are in tension here, and the tension is the finding.

### The ratio is an assumption, so it is swept

5:1 is a documented choice, not a measurement. Neither dataset records recovery rates or
interest margins, so any ratio claiming to be derived would be an assumption in a
measurement's clothing. The sensitivity analysis shows how much of the result is the data and
how much is the assumption:

| Cost ratio | Threshold | Approval rate | Recall (calibration) |
|---|---|---|---|
| 1:1 | 0.5368 | 0.9033 | 0.3135 |
| 2:1 | 0.3059 | 0.8005 | 0.5177 |
| 5:1 | 0.1899 | 0.6168 | 0.7136 |
| 10:1 | 0.1081 | 0.2688 | 0.9194 |
| 20:1 | 0.0726 | 0.0703 | 0.9902 |

Taiwan T0. The 1:1 row is the check worth reading: symmetric costs select 0.5368, almost
exactly the 0.5 the project used to inherit. That is what 0.5 was quietly assuming all along,
and it is evidence the selector does what it claims rather than merely producing a lower
number.

The served API applies the chosen threshold, not 0.5, and reports which one it used in
`threshold_basis`. A threshold that is computed and then not applied would be decoration.

---

## Adverse-action reason codes

Regulation B (12 CFR 1002.9) requires a lender to state the principal reasons for a decline.
A score without them is not a lending decision anyone can send. `POST /predict` with
`include_reasons=true` returns up to four, ranked by absolute SHAP contribution:

```
decision=decline  P(default)=0.6549  threshold=0.5000
  1. status_0    shap=+0.0567  status_0 applied to this application and pushed the risk score higher
  2. status_3    shap=+0.0441  status_3 did not apply to this application, and its absence
                               pushed the risk score higher
  3. duration    shap=+0.0210  duration pushed the risk score higher
  4. amount      shap=+0.0198  amount pushed the risk score higher
```

The second line is the part worth noticing. `status_3` is a one-hot column the applicant does
not hold, and its absence is genuinely what raised the score: not having the strongest
checking-account status is a real reason to decline. Phrasing it as "status 3 pushed the risk
score higher" would tell the applicant they were declined for a status they never had, which
misstates a principal reason. The distinction is enforced by a test, not by convention.

Attributions are computed on the encoded matrix against the classifier step, so an explanation
cannot disagree with the prediction it accompanies. Approvals return no reason codes, because
ECOA requires them for adverse action and returning them on an approval invites reading them as
an eligibility explanation. Cost is 0.0231 s per decision against 0.0043 s without. Feature
names are encoded column names, not applicant-facing prose; that mapping is a product step this
repository does not take. Full contract in [MODEL_CARD.md](MODEL_CARD.md).

---

## Repository layout

```
src/
  data/registry.py        per-dataset column roles, protected directions, provenance
  data/splits.py          the one seeded, fingerprinted three-way split
  preprocessing/          registry-driven encoder, Kamiran-Calders reweighing
  training/search.py      Optuna search over four model families
  evaluation/             group fairness metrics, bootstrap intervals, SHAP reason codes
  evaluation/operating_point.py  cost-based threshold selection and its sensitivity sweep
  pipelines/tracks.py     the five tracks, sharing one evaluation path
  serving/predictor.py    the single inference path
api/                      FastAPI service (no auth; see below)
app.py                    Streamlit demo, calls the same predictor
scripts/
  run_comparison.py       runs the tracks and records results
  generate_report_assets.py  produces every table and figure from the artifact
  verify_german_encoding.py  recovers and checks the dataset's own encoding
  backfill_shap_background.py  adds SHAP background to artifacts fitted before it existed
notebooks/
  01_eda_bias_audit.ipynb  dataset-level bias, and where the processed CSV comes from
  prototype_tabfm_comparison.ipynb  foundation-model prototype, not a project result
  tabfm_colab_stage.ipynb  Colab-only staging for the GPU scoring step
reports/track_comparison.json  the only permitted source of a published number
MODEL_CARD.md             intended use, limitations, legal position
```

The notebooks are checked by `tests/unit/test_notebooks.py`, which fails on a machine-specific
path, an import of a deleted module or of a package the project does not install, a reference to
a deleted file, or a stored error output. All four checks were added because the committed
notebooks violated them: one loaded `/home/aswani/automl/...`, one imported `aif360` which
`requirements.txt` deliberately excludes, and one shipped a `ModuleNotFoundError` traceback
under the title "Complete Pipeline Walkthrough". That notebook is deleted; it demonstrated the
retired pipeline and described a composite fairness objective that never existed in the code.

---

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt -r requirements-dev.txt

pytest -m "unit or integration"                      # 235 tests
python scripts/run_comparison.py --dataset german_credit
python scripts/run_comparison.py --dataset german_credit --tracks T4   # recorded TabFM scores
python scripts/generate_report_assets.py

uvicorn api.main:app --reload                        # API at :8000/docs
streamlit run app.py                                 # demo
```

The German Credit defaults reproduce the published numbers: seed 42, 60 Optuna trials, 5-fold
CV, 2,000 bootstrap replicates, about seven minutes for four tracks. **Taiwan must be run at its
own recorded budget** or its numbers will not match what is published here:

```bash
python scripts/run_comparison.py --dataset taiwan_credit \
    --trials 20 --cv-folds 3 --bootstrap 1000
```

Re-running both datasets and diffing the artifact against the committed one gives zero drift
across all nine runs, which is the check behind the claim that these numbers are reproducible
rather than merely recorded.

T4 needs no GPU here: it reads the probabilities recorded in `notebooks/exchange/` from the one
GPU run, and refuses them if their row identifiers are not exactly this split's test block.

Docker: `docker-compose up --build -d`.

---

## Security

The API has **no authentication**. Any caller who can reach it can score an applicant. It must
not be exposed publicly without an auth layer in front of it. This is stated in the service
description and in the module comments rather than being left for a reader to discover.

---

## Data

- **German Credit** (UCI Statlog, 1,000 rows). Retained as the small canonical benchmark and as
  the audit trail for the metric defect. Its integer encoding is recovered from the raw file and
  asserted in a test, because whether a column may be treated as ordinal depends on it.
- **Taiwan credit card default** (UCI dataset 350, 30,000 rows). Added as the mid-scale primary
  because it has both a real repayment outcome and explicit protected attributes. HMDA was
  rejected: it records application decisions but no repayment outcome, so a credit risk model
  cannot be trained on it.

---

## Limitations

Stated in full in [MODEL_CARD.md](MODEL_CARD.md). The short version: this is a portfolio
project, not a lending system.

- **The cost ratio is an assumption.** 5:1 is a stated choice, swept from 1:1 to 20:1, not a
  quantity either dataset supports measuring.
- **Choosing the operating point costs fairness**, and this project raises that trade without
  resolving it. Resolving it needs a policy input a portfolio repository is not entitled to
  invent.
- **German Credit's test block is too small for any fairness claim.** 62 women; every
  disparate-impact interval spans the 0.8 line.
- **T4 is German Credit only**, and it carries no operating point: the offline GPU run produced
  test-block predictions, and selecting a threshold needs calibration-block scores.
- **No out-of-time validation, and neither dataset can support one.** German Credit is a
  1973-75 snapshot; Taiwan is a single window (April to September 2005 history, October 2005
  default), so every row shares one period. This is a property of the data, not a step that was
  skipped, and a temporal claim here would have to be invented.
- **Neither model is calibrated for a real portfolio.** Default rates in these datasets have no
  guaranteed relationship to any current population.

---

## Author

**Aswani Sahoo** — [GitHub](https://github.com/AswaniSahoo) · [LinkedIn](https://linkedin.com/in/aswani-sahoo)

MIT licensed. See [LICENSE](LICENSE).
