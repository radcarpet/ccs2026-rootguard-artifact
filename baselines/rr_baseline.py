"""
Randomized Response Baseline (based on USENIX Security 2019)

Reference: Murakami & Kawamoto, "Utility-Optimized Local Differential Privacy
Mechanisms for Distribution Estimation", USENIX Security 2019.

Discretizes each measurement into B bins across its medical domain, then
applies standard ε-LDP Randomized Response: reports the true bin with
probability e^ε / (e^ε + B - 1) and a uniformly random bin otherwise.
The bin center is returned as the sanitized numeric value.

All values are treated as equally sensitive (no ULDP distinction).
Applied per non-classification node with uniform budget — no DAG awareness.

Privacy guarantee: Pure ε-LDP per node (basic composition across nodes).
"""

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


def randomized_response(true_bin, epsilon, num_bins):
    """
    Standard ε-LDP Randomized Response for categorical data.

    Reports the true bin with probability p = e^ε / (e^ε + B - 1),
    and a uniformly random bin from [0, B-1] with probability 1 - p.

    Args:
        true_bin: Index of the true bin (int in [0, num_bins-1])
        epsilon: Privacy parameter
        num_bins: Total number of bins B

    Returns:
        Reported bin index (int)
    """
    exp_eps = np.exp(min(epsilon, 500))  # cap to avoid overflow
    p_true = exp_eps / (exp_eps + num_bins - 1)

    if np.random.random() < p_true:
        return true_bin
    else:
        return np.random.randint(0, num_bins)


def sanitize_randomized_response(samples, nodes, edges, bi_lipschitz_constants,
                                 epsilon, num_bins=20):
    """
    Sanitize all non-classification nodes via discretized Randomized Response.

    Each node gets uniform budget ε. Values are discretized into num_bins
    bins across their medical domain, perturbed via RR, then mapped back
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

            # Discretize → RR → bin center
            true_bin = _discretize(true_val, domain_lower, domain_upper, num_bins)
            reported_bin = randomized_response(true_bin, epsilon, num_bins)
            sanitized_val = _bin_center(reported_bin, domain_lower, domain_upper,
                                        num_bins)

            sanitized[node] = round(float(sanitized_val), ndigits=4)
            budgets.append(epsilon)

        all_values.append(sanitized)
        epsilon_blocks.append(budgets)

    return all_values, epsilon_blocks
