| Track | Intervention | Stage | Deployable | ROC-AUC | Balanced Accuracy | Disparate Impact | Stat. Parity Diff. |
|---|---|---|---|---|---|---|---|
| T0 | none | control | yes | 0.7902 [0.7755, 0.8037] | 0.6629 [0.6500, 0.6756] | 0.9767 [0.9594, 0.9934] | -0.0209 [-0.0366, -0.0060] |
| T1 | Kamiran-Calders reweighing | pre-processing | yes | 0.7898 [0.7754, 0.8032] | 0.6625 [0.6494, 0.6756] | 0.9779 [0.9615, 0.9940] | -0.0198 [-0.0348, -0.0054] |
| T2 | ExponentiatedGradient, demographic parity | in-processing | yes | 0.6589 [0.6456, 0.6712] [*] | 0.6589 [0.6456, 0.6712] | 0.9800 [0.9636, 0.9961] | -0.0180 [-0.0327, -0.0035] |
| T3 | group-specific thresholds | post-processing | no | 0.7902 [0.7755, 0.8037] | 0.7220 [0.7082, 0.7352] | 0.9425 [0.9137, 0.9712] | -0.0426 [-0.0649, -0.0210] |

[*] T2 emits a decision, not a graded score. Its ROC-AUC therefore equals its balanced accuracy by construction, and its Brier score is computed on a two-valued score, so neither is comparable with the other tracks' ranking or calibration quality.
