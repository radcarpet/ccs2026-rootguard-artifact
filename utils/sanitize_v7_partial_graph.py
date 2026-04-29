"""
Partial-knowledge variant of V7-abs sanitization.

At knowledge_fraction=1.0: identical to sanitize_dep_aware_v7 (full Preempt++).
At knowledge_fraction=0.0: all derived nodes treated as independent roots (≈Vanilla).
Intermediate: a fraction of derived nodes are "forgotten" and noised independently.
"""

import itertools
import math
import random

import numpy as np

from preempt import sanitizer
from preempt.sanitizer import TEMPLATE_DOMAINS
from utils.utils import (
    get_topological_order,
    compute_worst_case_sensitivities,
    optimal_budget_allocation_exact,
)
from utils.med_domain.all_templates import template_expressions

# Formula dependencies (same as compute_leakage_table.py)
FORMULA_DEPS = {
    "mcv": ["hct", "rbc"], "mch": ["hb", "rbc"], "mchc": ["hb", "hct"],
    "fib4_prod": ["age", "ast"], "fib4_denom": ["plt", "alt"],
    "fib4": ["fib4_prod", "fib4_denom"],
    "non_hdl": ["tc", "hdl"], "ldl": ["tc", "hdl", "tg"],
    "aip": ["tg", "hdl"],
    "bmi": ["wt", "ht"], "wthr": ["waist", "ht"],
    "conicity": ["waist", "wt", "ht"],
    "pp": ["sbp", "dbp"], "map": ["dbp", "pp"],
    "mbp": ["sbp", "dbp"], "ppi": ["pp", "sbp"],
    "tyg_prod": ["tg", "glu"], "tyg_half": ["tyg_prod"],
    "tyg": ["tyg_half"],
    "homa_prod": ["glu", "ins"], "homa": ["homa_prod"],
    "nlr_sum": ["neu", "lym"], "nlr_diff": ["neu", "lym"],
    "nlr": ["neu", "lym"],
}

SKIP_NODES = {"anemia_class", "fib4_risk", "aip_risk", "ci_risk",
              "ppi_risk", "tyg_class", "homa_class", "nlr_class"}


def compute_intermediate_domain(node, all_domains):
    """
    Compute domain [min, max] for an intermediate node by evaluating its
    expression at all corners of its parents' domains.
    """
    deps = FORMULA_DEPS.get(node, [])
    if not deps or node not in template_expressions:
        return all_domains.get(node, (0, 1))

    # Collect parent domains (recursively computed if needed)
    parent_bounds = []
    for p in deps:
        if p in all_domains:
            parent_bounds.append(all_domains[p])
        else:
            # Parent is also intermediate — compute its domain first
            pd = compute_intermediate_domain(p, all_domains)
            all_domains[p] = pd
            parent_bounds.append(pd)

    # Evaluate at all corner combinations
    corners = list(itertools.product(*parent_bounds))
    vals = []
    for corner in corners:
        values = dict(zip(deps, corner))
        # Also include any already-known domains as scalar values for eval
        eval_ctx = dict(all_domains)  # won't be used directly, but safe
        eval_ctx.update(values)
        try:
            v = eval(template_expressions[node],
                     {"np": np, "abs": abs, "math": math}, values)
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass

    if vals:
        return (min(vals), max(vals))
    return all_domains.get(node, (0, 1))


def sanitize_v7_partial_graph(samples, nodes, edges, expressions,
                               bi_lipschitz_constants, epsilon,
                               target_keys, forget_nodes=None,
                               uniform_budget=False,
                               epsilon_min=0.005, epsilon_max=None,
                               mode="abs", num_candidates=1000,
                               template_name=None):
    """
    Dependency-aware sanitization with partial graph knowledge.

    Like sanitize_dep_aware_v7, but nodes in `forget_nodes` are treated
    as independent roots (noised separately) rather than derived via
    post-processing.

    Args:
        forget_nodes: Set of node names to treat as independent roots.
                      If None, all derived nodes are known (= full Preempt++).
        uniform_budget: If True, each root gets total_budget/n_roots
                        (uniform allocation). If False, uses the optimizer.
        template_name: Template name for per-template domain lookup.
        (other args same as sanitize_dep_aware_v7)

    Returns: (all_values, epsilon_blocks, all_allocation_info)
    """
    if forget_nodes is None:
        forget_nodes = set()
    else:
        forget_nodes = set(forget_nodes)

    san = sanitizer.Sanitizer()
    epsilon_blocks = []
    all_values = []
    all_allocation_info = []
    _wc_sensitivity_cache = {}

    # Get per-template domains and compute domains for forgotten intermediates
    tmpl_domains = {}
    if template_name and template_name in TEMPLATE_DOMAINS:
        tmpl_domains = dict(TEMPLATE_DOMAINS[template_name])

    # Precompute domains for forgotten nodes (they need exponential mechanism bounds)
    for node in forget_nodes:
        if node not in tmpl_domains:
            tmpl_domains[node] = compute_intermediate_domain(node, tmpl_domains)

    for sample in samples:
        topo_order, gt_values = get_topological_order(sample, edges)
        ground_truth = {k: v for k, v in gt_values.items()}
        sanitized_values = {}
        actual_budgets = {}

        # Identify root and derived nodes
        root_nodes = []
        derived_nodes = []
        for node in topo_order:
            if node not in nodes:
                sanitized_values[node] = ground_truth[node]
                actual_budgets[node] = 0
                continue
            prevs = nodes[node]
            if prevs:
                prevs = [p for p in prevs if p in ground_truth]
            if not prevs:
                root_nodes.append(node)
            else:
                derived_nodes.append(node)

        # Move forgotten nodes from derived to root
        for node in list(derived_nodes):
            if node in forget_nodes and node not in SKIP_NODES:
                derived_nodes.remove(node)
                root_nodes.append(node)

        # Find target node
        target_node = None
        for tname, tnode in target_keys.items():
            if tnode in ground_truth and tnode not in SKIP_NODES:
                target_node = tnode
                break

        # Total budget = number of non-skip nodes × ε
        n_nodes = len([n for n in topo_order if n not in SKIP_NODES])
        total_budget = n_nodes * epsilon

        if uniform_budget or not target_node or len(root_nodes) == 0:
            # Uniform: each root gets total_budget / n_roots
            eps_per_root = total_budget / max(len(root_nodes), 1)
            allocation = {r: eps_per_root for r in root_nodes}
            sensitivities = {r: 0.0 for r in root_nodes}
        else:
            # Worst-case sensitivities: data-independent (same for all patients)
            # Merge template-specific domains with global defaults
            merged_domains = dict(sanitizer.MEDICAL_DOMAINS)
            merged_domains.update(tmpl_domains)
            cache_key = (target_node, frozenset(root_nodes))
            if cache_key not in _wc_sensitivity_cache:
                _wc_sensitivity_cache[cache_key] = compute_worst_case_sensitivities(
                    target_node, root_nodes, expressions, SKIP_NODES,
                    topo_order, nodes, merged_domains
                )
            sensitivities = _wc_sensitivity_cache[cache_key]
            dw = {}
            for root in root_nodes:
                lo, hi = tmpl_domains.get(
                    root, sanitizer.MEDICAL_DOMAINS.get(
                        root, sanitizer.MEDICAL_DOMAINS["_default"]
                    )
                )
                dw[root] = hi - lo
            # true_indices=None → uses center of grid (data-independent)
            allocation = optimal_budget_allocation_exact(
                sensitivities, total_budget, epsilon_min, epsilon_max,
                domain_widths=dw, true_indices=None,
                num_candidates=num_candidates, mode=mode
            )

        # Noise root nodes (including forgotten derived nodes)
        for root in root_nodes:
            root_eps = allocation.get(root, epsilon)
            lo, hi = tmpl_domains.get(
                root, sanitizer.MEDICAL_DOMAINS.get(
                    root, sanitizer.MEDICAL_DOMAINS["_default"]
                )
            )
            noised_value = san.M_exponential_discrete(
                ground_truth[root], root_eps, lo, hi,
                num_candidates=num_candidates
            )
            sanitized_values[root] = round(float(noised_value), ndigits=4)
            actual_budgets[root] = root_eps

        # Post-process remaining known derived nodes
        for node in derived_nodes:
            if node in SKIP_NODES:
                sanitized_values[node] = ground_truth[node]
                actual_budgets[node] = 0
                continue
            if node in expressions:
                f_x_prime = eval(
                    expressions[node],
                    {"random": random, "round": round, "np": np, "math": math},
                    sanitized_values
                )
                if not np.isfinite(f_x_prime):
                    sanitized_values[node] = ground_truth[node]
                else:
                    sanitized_values[node] = f_x_prime
                actual_budgets[node] = 0
            else:
                sanitized_values[node] = ground_truth[node]
                actual_budgets[node] = 0

        all_values.append(sanitized_values)
        epsilon_blocks.append([actual_budgets.get(n, 0) for n in topo_order])
        all_allocation_info.append({
            "root_nodes": root_nodes,
            "forget_nodes": list(forget_nodes),
            "target_node": target_node,
            "allocation": {r: float(e) for r, e in allocation.items()},
            "total_budget": total_budget,
        })

    return all_values, epsilon_blocks, all_allocation_info
