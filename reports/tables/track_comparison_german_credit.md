| Track | Intervention | Stage | Deployable | ROC-AUC | Balanced Accuracy | Disparate Impact | Stat. Parity Diff. |
|---|---|---|---|---|---|---|---|
| T0 | none | control | yes | 0.8296 [0.7725, 0.8804] | 0.7357 [0.6679, 0.7988] | 0.7263 [0.5342, 0.9106] | -0.1884 [-0.3319, -0.0570] |
| T1 | Kamiran-Calders reweighing | pre-processing | yes | 0.8298 [0.7733, 0.8819] | 0.7143 [0.6464, 0.7774] | 0.6774 [0.4848, 0.8793] | -0.2151 [-0.3537, -0.0755] |
| T2 | ExponentiatedGradient, demographic parity | in-processing | yes | 0.7357 [0.6679, 0.7988] [*] | 0.7357 [0.6679, 0.7988] | 0.7263 [0.5342, 0.9106] | -0.1884 [-0.3319, -0.0570] |
| T3 | group-specific thresholds | post-processing | no | 0.8296 [0.7725, 0.8804] | 0.7333 [0.6738, 0.7881] | 0.6583 [0.4336, 0.9032] | -0.1758 [-0.3055, -0.0483] |

[*] T2 emits a decision, not a graded score. Its ROC-AUC therefore equals its balanced accuracy by construction, and its Brier score is computed on a two-valued score, so neither is comparable with the other tracks' ranking or calibration quality.
