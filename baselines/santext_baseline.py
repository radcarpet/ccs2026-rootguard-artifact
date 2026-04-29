"""
SanText Baseline (ACL Findings 2021)

Reference: Yue et al., "Differential Privacy for Text Analytics via
Natural Text Sanitization", ACL 2021 Findings.

Applies metric Local Differential Privacy (MLDP) at the token level using
the exponential mechanism over the full vocabulary. Each token t is replaced
by sampling t' with probability proportional to exp(-ε/2 · d(φ(t), φ(t')))
where φ maps tokens to embeddings and d is Euclidean distance.

Adaptation for medical data: tokenize the medical text paragraph, apply
SanText mechanism ONLY to numeric tokens (sensitive values), leave
non-numeric tokens (labels, units) unchanged. Total budget = n_nodes * ε,
distributed across the numeric BPE tokens only.

When numeric parsing fails, fall back to M_exponential_discrete.

Privacy guarantee: ε-metric-LDP per token (basic composition across
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

# Node ordering per template (node_name, decimal_places) — same as SBYW baseline
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


def _santext_mechanism(token_id, epsilon):
    """
    SanText exponential mechanism: sample a replacement token from the full
    vocabulary with probability ∝ exp(-ε/2 · d_euc(φ(t), φ(t'))).

    Args:
        token_id: Input token ID
        epsilon: Privacy parameter

    Returns:
        Replacement token ID
    """
    vocab_size = get_vocab_size()
    distances = euclidean_distances_from(token_id)

    # Log-probabilities: -ε/2 · d(t, t')
    log_probs = -epsilon / 2.0 * distances

    # Numerical stability
    log_probs -= np.max(log_probs)
    probs = np.exp(log_probs)
    probs /= probs.sum()

    return np.random.choice(vocab_size, p=probs)


def sanitize_santext(samples, nodes, edges, epsilon, template_target_keys,
                     template_name=None):
    """
    Sanitize medical values via SanText token-level MLDP.

    Budget allocation: total_budget = n_nodes * epsilon (matching vanilla).
    Only numeric BPE tokens are perturbed; non-numeric tokens (labels, units)
    are kept unchanged. The total budget is split uniformly across the
    numeric tokens: eps_per_token = total_budget / n_numeric_tokens.

    Pipeline per sample:
      1. Generate paragraph from ground truth values
      2. Tokenize and identify numeric tokens
      3. Apply SanText mechanism only to numeric tokens
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

        # Total budget matches vanilla: n_nodes * epsilon
        n_nodes = len(nodes_to_release)
        total_budget = n_nodes * epsilon

        # Step 1: Generate paragraph
        paragraph = sample_to_paragraph(sample, template_name)

        # Step 2: Tokenize and identify numeric tokens
        token_ids = tokenize_text(paragraph)
        is_numeric = identify_numeric_tokens(token_ids)
        n_numeric = sum(is_numeric)

        # Budget per numeric token (only numeric tokens consume budget)
        eps_per_token = total_budget / max(n_numeric, 1)

        # Step 3: Apply SanText only to numeric tokens; keep others unchanged
        sanitized_ids = []
        for i, tid in enumerate(token_ids):
            if is_numeric[i]:
                sanitized_ids.append(_santext_mechanism(tid, eps_per_token))
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
        print(f"    [SanText] Fallback rate: {fallback_count}/{total_values} "
              f"({rate:.1f}%)")

    return all_values, epsilon_blocks
