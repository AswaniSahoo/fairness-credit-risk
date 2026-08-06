| Track | Threshold | Approval rate | Recall at 0.5 | Recall at threshold | Disparate impact |
|---|---|---|---|---|---|
| T0 | 0.3606 | 0.3400 | 0.7000 | 0.9667 | 0.5261 |
| T1 | 0.3874 | 0.3950 | 0.7000 | 0.9500 | 0.7048 |

Threshold minimises expected cost at a 5:1 false-negative to false-positive ratio, fitted on the calibration block and reported on test.

| Cost ratio | Threshold | Approval rate | Recall (calibration) |
|---|---|---|---|
| 1:1 | 0.5365 | 0.7500 | 0.4833 |
| 2:1 | 0.4881 | 0.6350 | 0.6667 |
| 5:1 | 0.3606 | 0.3550 | 0.9333 |
| 10:1 | 0.3606 | 0.3550 | 0.9333 |
| 20:1 | 0.2790 | 0.1600 | 0.9833 |

Sensitivity for the control track. A ratio of 1:1 is what threshold 0.5 assumes, and it selects a threshold close to 0.5, which is the check that the selector is doing what it claims.
