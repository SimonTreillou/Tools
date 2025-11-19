import numpy as np
from itertools import combinations

# -------------- LIST OF FUNCTIONS IN THIS MODULE --------------
# variance_decomposition: Decompose variance of sum of terms into individual variances and covariances.
# ---------------------------------------------------------------

def variance_decomposition(terms):
    """
    Decompose the variance of the sum of multiple arrays into individual variances
    and pairwise covariances.

    Parameters
    ----------
    terms : dict[str, array_like]
        Mapping from a name to a numeric array. Each array is flattened and treated
        as a 1D sample over which variances/covariances are computed.

    Returns
    -------
    dict
        {
            "var": dict(name -> variance),
            "cov": dict((name_a, name_b) -> covariance),
            "sum_var": sum of individual variances,
            "sum_cov": 2 * sum of pairwise covariances (for full decomposition),
            "decomposition": sum_var + sum_cov,
            "actual_var": variance of the elementwise sum of inputs,
            "error": decomposition - actual_var
        }
    """
    # Ensure inputs are numpy 1D arrays (flattened) so all grid points are included.
    X = {name: np.asarray(arr).ravel() for name, arr in terms.items()}

    # --- Individual variances (population variance, ddof=0) ---
    var = {name: np.var(x) for name, x in X.items()}
    # Sort variances descending for easier inspection
    var = dict(sorted(var.items(), key=lambda item: item[1], reverse=True))

    # --- Pairwise covariances ---
    cov = {}
    for a, b in combinations(X.keys(), 2):
        # np.cov returns a 2x2 covariance matrix for two 1D inputs; pick the off-diagonal.
        cov[(a, b)] = np.cov(X[a], X[b])[0, 1]
    # Sort covariances descending
    cov = dict(sorted(cov.items(), key=lambda item: item[1], reverse=True))

    # --- Full decomposition (sum of variances + 2 * sum of unique covariances) ---
    sum_of_vars = sum(var.values())
    sum_of_covs = 2 * sum(cov.values())
    total_decomposition = sum_of_vars + sum_of_covs

    # --- Actual variance of the elementwise sum of the terms ---
    # Stack as rows (n_terms, n_points) and sum across rows to get the elementwise sum.
    Xsum = np.sum(np.vstack(list(X.values())), axis=0)
    actual_variance = np.var(Xsum)

    return {
        "var": var,
        "cov": cov,
        "sum_var": sum_of_vars,
        "sum_cov": sum_of_covs,
        "decomposition": total_decomposition,
        "actual_var": actual_variance,
        "error": total_decomposition - actual_variance,
    }
