"""
CusText Baseline (ACL Findings 2023)

Reference: Lu et al., "A Customized Text Sanitization Mechanism with
Differential Privacy", ACL 2023 Findings.

Assigns each token a customized output set of K semantically similar tokens.
The exponential mechanism samples a replacement from this restricted set,
with score function u(x,y) based on normalized similarity. This reduces the
sampling space from the full vocabulary to K candidates, improving the
privacy-utility tradeoff compared to SanText.

Adaptation for medical data: same text pipeline as SanText, but with K=20
customized output sets per token. Only numeric tokens are perturbed;
total budget = n_nodes * ε, distributed across numeric tokens.

Privacy guarantee: Pure ε-DP per token (basic composition across
the sanitized numeric tokens).
"""

import numpy as np

from preempt.sanitizer import MEDICAL_DOMAINS, Sanitizer
from utils.utils import get_topological_order
from baselines.dp_gtr_baseline import sample_to_paragraph
from baselines.text_dp_common import (
    tokenize_text, decode_tokens, euclidean_distances_from,
    get_vocab_size, extract_numeric_values_from_text,
    identify_numeric_tokens,
)


_SKIP_NODES = frozenset([
    "anemia_class", "fib4_risk", "aip_risk", "ci_risk",
    "ppi_risk", "tyg_class", "homa_class", "nlr_class",
])

_TEMPLATE_VALUE_ORDER = {
    "ANEMIA": [
        ("hb", 1), ("hct", 1), ("rbc", 2), ("mcv", 2), ("mch", 2), ("mchc", 2),
    ],
    "AIP": [
        ("tc", 1), ("hdl", 1), ("non_hdl", 1), ("tg", 1), ("ldl", 1), ("aip", 4),
    ],
    "CONICITY": [
        ("wt", 1), ("ht", 2), ("bmi", 2), ("waist", 2), ("wthr", 2),
        ("conicity", 2),
    ],
    "VASCULAR": [
        ("sbp", 1), ("dbp", 1), ("pp", 1), ("map", 2), ("mbp", 2), ("ppi", 4),
    ],
    "FIB4": [
        ("age", 0), ("ast", 1), ("alt", 1), ("plt", 1), ("fib4_prod", 2),
        ("fib4_denom", 2), ("fib4", 2),
    ],
    "TYG": [
        ("tg", 1), ("glu", 1), ("tyg_prod", 2), ("tyg_half", 2), ("tyg", 4),
    ],
    "HOMA": [
        ("glu", 1), ("ins", 2), ("homa_prod", 2), ("homa", 4),
    ],
    "NLR": [
        ("neu", 1), ("lym", 1), ("nlr_sum", 2), ("nlr_diff", 2), ("nlr", 4),
    ],
}

# ---------------------------------------------------------------------------
# Customized output set construction (cached per session)
# ---------------------------------------------------------------------------

_cached_output_sets = {}  # dict: token_id → array of K nearest token IDs
_K = 20  # output set size (matching CusText paper default)


def _get_output_set(token_id):
    """Get the customized output set for a token (lazy, cached)."""
    if token_id in _cached_output_sets:
        return _cached_output_sets[token_id]

    # Compute K-nearest neighbors for this token
    distances = euclidean_distances_from(token_id)
    nearest_k = np.argpartition(distances, _K)[:_K]
    _cached_output_sets[token_id] = nearest_k

    return nearest_k


def _custext_mechanism(token_id, epsilon):
    """
    CusText exponential mechanism with customized output sets.

    For input token t, samples t' from the K-nearest output set with
    probability ∝ exp(ε · u(t, t') / (2Δu)) where:
      u(t, t') = 1 - d_norm(t, t')  (normalized Euclidean distance)
      Δu = 1  (sensitivity after normalization to [0, 1])

    Args:
        token_id: Input token ID
        epsilon: Privacy parameter

    Returns:
        Replacement token ID from the customized output set
    """
    output_set = _get_output_set(token_id)

    # Compute distances to output set members only
    distances = euclidean_distances_from(token_id)
    set_distances = distances[output_set]

    # Normalize distances to [0, 1] within the output set
    d_max = set_distances.max()
    if d_max > 0:
        normalized = set_distances / d_max
    else:
        normalized = np.zeros_like(set_distances)

    # Score: u(t, t') = 1 - normalized_distance, Δu = 1
    scores = 1.0 - normalized

    # Exponential mechanism: sample ∝ exp(ε · score / 2)
    log_probs = epsilon * scores / 2.0
    log_probs -= np.max(log_probs)
    probs = np.exp(log_probs)
    probs /= probs.sum()

    idx = np.random.choice(len(output_set), p=probs)
    return int(output_set[idx])


def sanitize_custext(samples, nodes, edges, epsilon, template_target_keys,
                     template_name=None):
    """
    Sanitize medical values via CusText customized-output-set DP.

    Budget allocation: total_budget = n_nodes * epsilon (matching vanilla).
    Only numeric BPE tokens are perturbed; non-numeric tokens kept unchanged.
    Budget split uniformly across numeric tokens.

    Pipeline per sample:
      1. Generate paragraph from ground truth values
      2. Tokenize and identify numeric tokens
      3. Apply CusText mechanism only to numeric tokens (K=20 output set)
      4. Decode sanitized text and parse numeric values
      5. Fall back to M_exponential_discrete for unparseable values

    Returns:
        (all_values, epsilon_blocks) matching vanilla format
    """
    san = Sanitizer()
    epsilon_blocks = []
    all_values = []
    fallback_count = 0
    total_values = 0

    value_order = _TEMPLATE_VALUE_ORDER.get(template_name, [])

    for sample in samples:
        topo_order, values = get_topological_order(sample, edges)
        nodes_to_release = [n for n in topo_order if n not in _SKIP_NODES]

        n_nodes = len(nodes_to_release)
        total_budget = n_nodes * epsilon

        # Step 1: Generate paragraph
        paragraph = sample_to_paragraph(sample, template_name)

        # Step 2: Tokenize and identify numeric tokens
        token_ids = tokenize_text(paragraph)
        is_numeric = identify_numeric_tokens(token_ids)
        n_numeric = sum(is_numeric)
        eps_per_token = total_budget / max(n_numeric, 1)

        # Step 3: Apply CusText only to numeric tokens
        sanitized_ids = []
        for i, tid in enumerate(token_ids):
            if is_numeric[i]:
                sanitized_ids.append(_custext_mechanism(tid, eps_per_token))
            else:
                sanitized_ids.append(tid)

        # Step 4: Decode and parse
        sanitized_text = decode_tokens(sanitized_ids)
        parsed, failed = extract_numeric_values_from_text(
            sanitized_text, value_order)

        # Step 5: Build sanitized values dict with fallback
        sanitized = {}
        budgets = []
        for node in nodes_to_release:
            total_values += 1
            if node in parsed:
                domain_lower, domain_upper = MEDICAL_DOMAINS.get(
                    node, MEDICAL_DOMAINS["_default"])
                val = np.clip(parsed[node], domain_lower, domain_upper)
                sanitized[node] = round(float(val), ndigits=4)
            else:
                fallback_count += 1
                domain_lower, domain_upper = MEDICAL_DOMAINS.get(
                    node, MEDICAL_DOMAINS["_default"])
                noised = san.M_exponential_discrete(
                    float(values[node]), epsilon, domain_lower, domain_upper)
                sanitized[node] = round(float(noised), ndigits=4)
            budgets.append(epsilon)

        all_values.append(sanitized)
        epsilon_blocks.append(budgets)

    if total_values > 0:
        rate = fallback_count / total_values * 100
        print(f"    [CusText] Fallback rate: {fallback_count}/{total_values} "
              f"({rate:.1f}%)")

    return all_values, epsilon_blocks
