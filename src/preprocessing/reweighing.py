"""Kamiran and Calders (2012) reweighing.

Reimplemented rather than calling ``aif360.algorithms.preprocessing.Reweighing`` because
aif360 0.5.0 cannot coexist with the numpy 2.x that TabFM requires. See decision C5 in
``.agents/plans/tabfm-integration.md``.

The weights are a closed form. For protected group ``a`` and label ``y``:

    w(a, y) = P(A = a) * P(Y = y) / P(A = a, Y = y)

A (group, outcome) cell that is under-represented relative to the product of its marginals
receives weight above 1, and an over-represented cell below 1. Total weight is conserved,
so the effective sample size does not change.

Measured caveat: on German Credit this changed only 4 of 200 test predictions and produced
no distinguishable fairness gain while distinguishably reducing balanced accuracy and F1.
The implementation is correct; the method is simply weak on that dataset. See the
prototype results in the project context.

Reference: Kamiran, F. and Calders, T. (2012). Data preprocessing techniques for
classification without discrimination. Knowledge and Information Systems 33(1).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def reweighing_weights(y_true: ArrayLike, groups: ArrayLike) -> NDArray[np.float64]:
    """Compute per-sample reweighing weights.

    Args:
        y_true: Training labels.
        groups: Protected attribute value per row. Any encoding; every distinct value is
            treated as its own group, so this works for both the 0/1 encoding of German
            Credit and the 1/2 encoding of the Taiwan dataset.

    Returns:
        Weights aligned to the input rows, summing to the number of rows.

    Raises:
        ValueError: If the inputs differ in length.
    """
    y_arr = np.asarray(y_true).ravel()
    group_arr = np.asarray(groups).ravel()

    if y_arr.size != group_arr.size:
        raise ValueError(
            f"y_true and groups must be the same length; got {y_arr.size}, {group_arr.size}"
        )

    n_rows = y_arr.size
    weights = np.ones(n_rows, dtype=np.float64)

    for group_value in np.unique(group_arr):
        in_group = group_arr == group_value
        n_group = int(in_group.sum())
        for label in np.unique(y_arr):
            has_label = y_arr == label
            cell = in_group & has_label
            n_cell = int(cell.sum())
            if n_cell == 0:
                # An empty cell has no rows to weight; leaving it is correct, and
                # dividing by zero here would silently produce inf weights.
                continue
            weights[cell] = (n_group * int(has_label.sum())) / (n_rows * n_cell)

    return weights
