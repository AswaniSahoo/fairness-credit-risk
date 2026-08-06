| Track | Threshold | Approval rate | Recall at 0.5 | Recall at threshold | Disparate impact |
|---|---|---|---|---|---|
| T0 | 0.1899 | 0.6063 | 0.3647 | 0.7272 | 0.9374 |
| T1 | 0.1879 | 0.5995 | 0.3640 | 0.7340 | 0.9414 |

Threshold minimises expected cost at a 5:1 false-negative to false-positive ratio, fitted on the calibration block and reported on test.

| Cost ratio | Threshold | Approval rate | Recall (calibration) |
|---|---|---|---|
| 1:1 | 0.5368 | 0.9033 | 0.3135 |
| 2:1 | 0.3059 | 0.8005 | 0.5177 |
| 5:1 | 0.1899 | 0.6168 | 0.7136 |
| 10:1 | 0.1081 | 0.2688 | 0.9194 |
| 20:1 | 0.0726 | 0.0703 | 0.9902 |

Sensitivity for the control track. A ratio of 1:1 is what threshold 0.5 assumes, and it selects a threshold close to 0.5, which is the check that the selector is doing what it claims.
