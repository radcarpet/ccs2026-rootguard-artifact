"""
Subset Selection Baseline (based on USENIX Security 2025)

Reference: He et al., "Addressing Sensitivity Distinction in Local
Differential Privacy", USENIX Security 2025.

Discretizes each measurement into B bins across its medical domain, then
applies the Subset Selection (SS) mechanism: selects a random subset of d
bins where the true bin is included with probability proportional to e^ε.
Returns a uniformly random element from the selected subset.

All values are treated as equally sensitive (no ULDP distinction), so the
mechanism reduces to standard SS.  Applied per non-classification node
with uniform budget — no DAG awareness.

Privacy guarantee: Pure ε-LDP per node (basic composition across nodes).
"""

import math
import numpy as np

from preempt.sanitizer import MEDICAL_DOMAINS
from utils.utils import get_topological_order


_SKIP_NODES = frozenset([
    "anemia_class", "fib4_risk", "aip_risk", "ci_risk",
    "ppi_risk", "tyg_class", "homa_class", "nlr_class",
])


def _discretize(value, domain_lower, domain_upper, num_bins):
    """Map a continuous value to the nearest bin index in [0, num_bins-1]."""
    if domain_upper <= domain_lower:
        return 0
    frac = (value - domain_lower) / (domain_upper - domain_lower)
    frac = np.clip(frac, 0.0, 1.0)
    idx = int(round(frac * (num_bins - 1)))
    return np.clip(idx, 0, num_bins - 1)


def _bin_center(idx, domain_lower, domain_upper, num_bins):
    """Map a bin index back to the center value in the original domain."""
    if num_bins <= 1:
        return (domain_lower + domain_upper) / 2.0
    return domain_lower + idx * (domain_upper - domain_lower) / (num_bins - 1)


def _optimal_subset_size(num_bins, epsilon):
    """
    Compute the optimal subset size d for Subset Selection.

    From the SS literature, the optimal d satisfies:
        d = max(1, min(B, floor(B / (e^ε + 1)) + 1))

    This balances the trade-off: smaller d → more privacy but less utility.
    When ε is large, d → 1 (deterministic). When ε → 0, d → B/2.
    """
    exp_eps = math.exp(min(epsilon, 500))
    d = max(1, min(num_bins, int(num_bins / (exp_eps + 1)) + 1))
    return d


def subset_selection(true_bin, epsilon, num_bins):
    """
    Subset Selection mechanism for ε-LDP.

    Selects a subset S of size d from [0, B-1]:
      - If true_bin ∈ S: probability ∝ e^ε
      - If true_bin ∉ S: probability ∝ 1
    Then outputs a uniformly random element from S.

    Args:
        true_bin: Index of the true bin (int in [0, num_bins-1])
        epsilon: Privacy parameter
        num_bins: Total number of bins B

    Returns:
        Reported bin index (int)
    """
    d = _optimal_subset_size(num_bins, epsilon)
    exp_eps = math.exp(min(epsilon, 500))

    # Probability that the selected subset contains the true bin
    # P(true_bin ∈ S) = d·e^ε / (d·e^ε + B - d)
    p_include = (d * exp_eps) / (d * exp_eps + num_bins - d)

    all_bins = np.arange(num_bins)

    if np.random.random() < p_include:
        # Subset includes the true bin: pick d-1 others uniformly
        others = np.delete(all_bins, true_bin)
        if d - 1 > 0 and len(others) > 0:
            chosen = np.random.choice(others, size=min(d - 1, len(others)),
                                      replace=False)
            subset = np.append(chosen, true_bin)
        else:
            subset = np.array([true_bin])
    else:
        # Subset excludes the true bin: pick d from the remaining B-1 bins
        others = np.delete(all_bins, true_bin)
        subset = np.random.choice(others, size=min(d, len(others)),
                                  replace=False)

    # Output a uniformly random element from the subset
    return int(np.random.choice(subset))


def sanitize_subset_selection(samples, nodes, edges, bi_lipschitz_constants,
                              epsilon, num_bins=20):
    """
    Sanitize all non-classification nodes via discretized Subset Selection.

    Each node gets uniform budget ε. Values are discretized into num_bins
    bins across their medical domain, perturbed via SS, then mapped back
    to the bin center.

    Args:
        samples: List of sample dicts (same format as other sanitizers)
        nodes: Template node definitions
        edges: Template edge definitions
        bi_lipschitz_constants: Not used (kept for API compatibility)
        epsilon: Per-node privacy budget
        num_bins: Number of discretization bins (default 20)

    Returns:
        (all_values, epsilon_blocks) matching the vanilla sanitizer format
    """
    epsilon_blocks = []
    all_values = []

    for sample in samples:
        topo_order, values = get_topological_order(sample, edges)
        nodes_to_release = [n for n in topo_order if n not in _SKIP_NODES]
        sanitized = {}
        budgets = []

        for node in nodes_to_release:
            domain_lower, domain_upper = MEDICAL_DOMAINS.get(
                node, MEDICAL_DOMAINS["_default"]
            )
            true_val = float(values[node])

            # Discretize → SS → bin center
            true_bin = _discretize(true_val, domain_lower, domain_upper, num_bins)
            reported_bin = subset_selection(true_bin, epsilon, num_bins)
            sanitized_val = _bin_center(reported_bin, domain_lower, domain_upper,
                                        num_bins)

            sanitized[node] = round(float(sanitized_val), ndigits=4)
            budgets.append(epsilon)

        all_values.append(sanitized)
        epsilon_blocks.append(budgets)

    return all_values, epsilon_blocks
