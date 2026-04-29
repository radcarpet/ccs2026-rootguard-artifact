"""
InferDPT Baseline — 1D Numeric and Filtered (Embedding-Based) Variants

Two variants:

  InferDPT-N (sanitize_inferdpt): Adapts InferDPT's two-step mechanism
    (Laplace noise + exponential mechanism) to work directly on 1D numeric
    medical values. Every non-classification node is noised independently.

  InferDPT-F (sanitize_inferdpt_filtered): Operates at the BPE token level
    using text-embedding-ada-002 embeddings. Tokenizes values with RoBERTa,
    applies InferDPT's mechanism in embedding space per token, restricted
    to numeric-only tokens (filtered). Budget split across tokens.

Reference: InferDPT func.py lines 133-178 for the original text-level mechanism.
"""

import math
import os
import re
import random
import numpy as np

from preempt.sanitizer import MEDICAL_DOMAINS, Sanitizer
from utils.utils import get_topological_order


def inferdpt_rantext_1d(value, epsilon, domain_lower, domain_upper,
                        num_candidates=1000, mu_K=50):
    """
    Faithful RANTEXT adaptation for 1D numeric values on a discrete grid.

    Three-step mechanism preserving RANTEXT's structure:
      Step 1: Sample adjacency list size from positive-truncated Laplace.
          K_hat = max(1, round(|Lap(mu_K, 1/epsilon)|)), capped at m/2
      Step 2: Build adjacency list — K_hat nearest grid points to t,
          excluding t itself. Symmetric interval, boundary-aware.
      Step 3: Exponential mechanism over adjacency list:
          Pr[s] ∝ exp(-epsilon * |t - s| / 2)
          Sensitivity Δu = 1 (utility = negative index distance).
      Map back: x' = s * delta + ell

    Args:
        value: Original numeric value
        epsilon: Privacy budget
        domain_lower: Lower bound of the medical domain
        domain_upper: Upper bound of the medical domain
        num_candidates: Grid resolution (m = 1000)
        mu_K: Base neighborhood size in grid points

    Returns:
        Replacement value on the discrete grid within [ℓ, u]
    """
    domain_width = domain_upper - domain_lower
    if domain_width <= 0 or epsilon <= 0:
        return value

    m = num_candidates
    delta = domain_width / (m - 1)
    t = (value - domain_lower) / delta
    t_int = int(np.clip(round(t), 0, m - 1))

    # Step 1: Sample adjacency list size
    K_hat = int(max(1, round(abs(np.random.laplace(mu_K, 1.0 / epsilon)))))
    K_hat = min(K_hat, m // 2)

    # Step 2: Build adjacency list — K_hat nearest grid points, excluding t
    lo_idx = max(0, t_int - K_hat)
    hi_idx = min(m - 1, t_int + K_hat)
    adjacency = [s for s in range(lo_idx, hi_idx + 1) if s != t_int]

    if len(adjacency) == 0:
        # Edge case: only one grid point — return it
        return float(t_int * delta + domain_lower)

    # Step 3: Exponential mechanism over adjacency list
    adj_arr = np.array(adjacency, dtype=float)
    distances = np.abs(adj_arr - t_int)
    log_scores = -epsilon * distances / 2.0
    log_scores -= log_scores.max()  # numerical stability
    scores = np.exp(log_scores)
    probs = scores / scores.sum()

    s = int(np.random.choice(adj_arr, p=probs))
    return float(s * delta + domain_lower)


def sanitize_inferdpt(samples, nodes, edges, expressions, epsilon,
                      target_keys):
    """
    InferDPT baseline sanitization.

    Every non-classification node is noised independently using its ground
    truth value and MEDICAL_DOMAINS bounds. This mirrors InferDPT's
    token-level approach: each value is treated as an independent token
    with no knowledge of DAG structure.

    Args:
        samples: List of sample dictionaries with conversation turns
        nodes: Dictionary mapping node names to their parent dependencies
            (template_nodes)
        edges: List of (source, target) dependency edges (template_edges)
        expressions: Dictionary mapping node names to computation expressions
            (template_expressions)
        epsilon: Privacy budget
        target_keys: Dict mapping template name to target node name

    Returns:
        (all_values, epsilon_blocks) — same format as sanitize_vanilla()
    """
    epsilon_blocks = []
    all_values = []
    skip_nodes = ["anemia_class", "fib4_risk", "aip_risk",
                  "ci_risk", "ppi_risk", "tyg_class", "homa_class",
                  "nlr_class"]

    for sample in samples:
        topo_order, gt_values = get_topological_order(sample, edges)
        ground_truth = {k: v for k, v in gt_values.items()}
        sanitized_values = {}
        actual_budgets = {}

        for node in topo_order:
            if node in skip_nodes:
                # Classification nodes — pass through
                sanitized_values[node] = ground_truth[node]
                actual_budgets[node] = 0
                continue

            if node not in nodes:
                # Constant/auxiliary — pass through
                sanitized_values[node] = ground_truth[node]
                actual_budgets[node] = 0
                continue

            # Noise every node independently using its ground truth value
            value = ground_truth[node]
            lo, hi = MEDICAL_DOMAINS.get(node, MEDICAL_DOMAINS["_default"])

            replacement = inferdpt_rantext_1d(value, epsilon, lo, hi)

            sanitized_values[node] = replacement
            actual_budgets[node] = epsilon

        all_values.append(sanitized_values)
        epsilon_blocks.append(
            [actual_budgets.get(node, 0) for node in topo_order]
        )

    return all_values, epsilon_blocks


def sanitize_inferdpt_roots_only(samples, nodes, edges, expressions, epsilon,
                                 num_candidates=1000):
    """
    InferDPT Roots-Only: noise only root nodes with InferDPT's two-step
    mechanism (Laplace + exponential), then post-process derived nodes.

    Same total budget B = n_nodes * epsilon as InferDPT-All, but distributed
    uniformly across the k root nodes (each gets B/k). Derived nodes are
    computed via post-processing (free by the post-processing theorem).

    Args:
        samples: List of sample dictionaries with conversation turns
        nodes: Dictionary mapping node names to their parent dependencies
        edges: List of (source, target) dependency edges
        expressions: Dict mapping node names to computation expressions
        epsilon: Base privacy budget per node
        num_candidates: Grid resolution (must match PREEMPT's m=1000)

    Returns:
        (all_values, epsilon_blocks)
    """
    epsilon_blocks = []
    all_values = []
    skip_nodes = ["anemia_class", "fib4_risk", "aip_risk", "ci_risk",
                  "ppi_risk", "tyg_class", "homa_class", "nlr_class"]

    for sample in samples:
        topo_order, gt_values = get_topological_order(sample, edges)
        ground_truth = {k: v for k, v in gt_values.items()}
        sanitized_values = {}
        actual_budgets = {}

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

        # Total budget matches other methods
        n_nodes = len([n for n in topo_order if n not in skip_nodes])
        total_budget = n_nodes * epsilon

        # Uniform allocation across roots
        k = len(root_nodes)
        root_eps = total_budget / k if k > 0 else epsilon

        # Noise roots with InferDPT two-step mechanism
        for root in root_nodes:
            lo, hi = MEDICAL_DOMAINS.get(root, MEDICAL_DOMAINS["_default"])
            replacement = inferdpt_rantext_1d(
                ground_truth[root], root_eps, lo, hi,
                num_candidates=num_candidates)
            sanitized_values[root] = round(float(replacement), ndigits=4)
            actual_budgets[root] = root_eps

        # Post-process derived nodes from noised roots
        for node in derived_nodes:
            if node in skip_nodes:
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
                    prevs = nodes.get(node, [])
                    if prevs:
                        prevs = [p for p in prevs if p in ground_truth]
                    parent_vals = {p: sanitized_values.get(p) for p in prevs}
                    raise ValueError(
                        f"Node '{node}' produced non-finite value: {f_x_prime}. "
                        f"Expression: {expressions[node]}, Parents: {parent_vals}"
                    )
                sanitized_values[node] = f_x_prime
                actual_budgets[node] = 0
            else:
                raise ValueError(
                    f"Derived node '{node}' has no expression and is not skip"
                )

        all_values.append(sanitized_values)
        epsilon_blocks.append(
            [actual_budgets.get(node, 0) for node in topo_order]
        )

    return all_values, epsilon_blocks


def sanitize_inferdpt_roots_opt(samples, nodes, edges, expressions, epsilon,
                                target_keys, population_means,
                                num_candidates=1000, epsilon_min=0.005):
    """
    InferDPT Roots with P++'s optimal budget allocation.

    Same as sanitize_inferdpt_roots_only but replaces uniform B/k allocation
    with sensitivity-weighted allocation from optimal_budget_allocation_exact.
    """
    from utils.utils import (optimal_budget_allocation_exact,
                             compute_population_mean_sensitivities)
    from preempt import sanitizer

    epsilon_blocks = []
    all_values = []
    skip_nodes = ["anemia_class", "fib4_risk", "aip_risk", "ci_risk",
                  "ppi_risk", "tyg_class", "homa_class", "nlr_class"]
    _sens_cache = {}

    for sample in samples:
        topo_order, gt_values = get_topological_order(sample, edges)
        ground_truth = {k: v for k, v in gt_values.items()}
        sanitized_values = {}
        actual_budgets = {}

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

        # Find target node
        target_node = None
        for tname, tnode in target_keys.items():
            if tnode in ground_truth and tnode not in skip_nodes:
                target_node = tnode
                break

        n_nodes = len([n for n in topo_order if n not in skip_nodes])
        total_budget = n_nodes * epsilon

        if target_node and len(root_nodes) > 0:
            cache_key = (target_node, frozenset(root_nodes))
            if cache_key not in _sens_cache:
                _sens_cache[cache_key] = compute_population_mean_sensitivities(
                    target_node, root_nodes, expressions, skip_nodes,
                    topo_order, nodes, population_means)
            sensitivities = _sens_cache[cache_key]

            dw = {}
            for root in root_nodes:
                lo, hi = sanitizer.MEDICAL_DOMAINS.get(
                    root, sanitizer.MEDICAL_DOMAINS["_default"])
                dw[root] = hi - lo

            allocation = optimal_budget_allocation_exact(
                sensitivities, total_budget, epsilon_min,
                domain_widths=dw, num_candidates=num_candidates, mode="abs")
        else:
            allocation = {r: total_budget / len(root_nodes) for r in root_nodes}

        # Noise roots with InferDPT two-step mechanism
        for root in root_nodes:
            root_eps = allocation.get(root, epsilon)
            lo, hi = MEDICAL_DOMAINS.get(root, MEDICAL_DOMAINS["_default"])
            replacement = inferdpt_rantext_1d(
                ground_truth[root], root_eps, lo, hi,
                num_candidates=num_candidates)
            sanitized_values[root] = round(float(replacement), ndigits=4)
            actual_budgets[root] = root_eps

        # Post-process derived nodes
        for node in derived_nodes:
            if node in skip_nodes:
                sanitized_values[node] = ground_truth[node]
                actual_budgets[node] = 0
                continue
            if node in expressions:
                f_x_prime = eval(
                    expressions[node],
                    {"random": random, "round": round, "np": np, "math": math},
                    sanitized_values)
                if not np.isfinite(f_x_prime):
                    prevs = nodes.get(node, [])
                    if prevs:
                        prevs = [p for p in prevs if p in ground_truth]
                    parent_vals = {p: sanitized_values.get(p) for p in prevs}
                    raise ValueError(
                        f"Node '{node}' produced non-finite value: {f_x_prime}. "
                        f"Expression: {expressions[node]}, Parents: {parent_vals}")
                sanitized_values[node] = f_x_prime
                actual_budgets[node] = 0
            else:
                raise ValueError(
                    f"Derived node '{node}' has no expression and is not skip")

        all_values.append(sanitized_values)
        epsilon_blocks.append(
            [actual_budgets.get(node, 0) for node in topo_order])

    return all_values, epsilon_blocks


# ============================================================================
# InferDPT-F: Filtered, embedding-based variant
# ============================================================================

_SKIP_NODES = frozenset([
    "anemia_class", "fib4_risk", "aip_risk", "ci_risk",
    "ppi_risk", "tyg_class", "homa_class", "nlr_class",
    "const",
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

# Lazy-loaded state for InferDPT-F
_roberta_tokenizer = None
_ada_embeddings = None       # np.ndarray (n_numeric, 1536)
_ada_numeric_ids = None      # np.ndarray of int token IDs
_ada_id_to_idx = None        # dict: token_id → index into _ada_embeddings
_ada_per_dim_sens = None     # np.ndarray (1536,) per-dimension sensitivity
_ADA_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..",
                                "data", "ada002_numeric_embeddings.npz")


def _load_roberta_tokenizer():
    """Lazy-load RoBERTa tokenizer and build numeric mask."""
    global _roberta_tokenizer
    if _roberta_tokenizer is not None:
        return

    from transformers import AutoTokenizer
    _roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    print("    [InferDPT-F] Loaded RoBERTa tokenizer", flush=True)


def _load_ada_embeddings():
    """Load or compute ada-002 embeddings for all numeric RoBERTa tokens.

    Embeddings are cached to disk to avoid repeated API calls.
    """
    global _ada_embeddings, _ada_numeric_ids, _ada_id_to_idx, _ada_per_dim_sens

    if _ada_embeddings is not None:
        return

    _load_roberta_tokenizer()

    cache_path = os.path.abspath(_ADA_CACHE_PATH)

    if os.path.exists(cache_path):
        print(f"    [InferDPT-F] Loading cached ada-002 embeddings from {cache_path}",
              flush=True)
        data = np.load(cache_path)
        cached_ids = data["token_ids"]
        cached_embs = data["embeddings"]

        # Filter to current (tighter) numeric mask — cache may contain
        # garbage tokens from an older, looser filter.
        from baselines.text_dp_common import build_roberta_numeric_mask
        mask = build_roberta_numeric_mask()
        keep = np.array([mask[tid] for tid in cached_ids])
        _ada_numeric_ids = cached_ids[keep]
        _ada_embeddings = cached_embs[keep]
        print(f"    [InferDPT-F] Filtered {keep.sum()}/{len(cached_ids)} tokens "
              f"from cache", flush=True)
    else:
        print("    [InferDPT-F] Computing ada-002 embeddings for numeric tokens...",
              flush=True)

        # Build numeric token list from RoBERTa vocab
        from baselines.text_dp_common import build_roberta_numeric_mask
        mask = build_roberta_numeric_mask()
        _ada_numeric_ids = np.where(mask)[0]

        # Decode each numeric token
        decoded_tokens = []
        for tid in _ada_numeric_ids:
            decoded = _roberta_tokenizer.decode([tid]).strip()
            decoded_tokens.append(decoded)

        print(f"    [InferDPT-F] Embedding {len(decoded_tokens)} numeric tokens "
              f"via Azure ada-002...", flush=True)

        # Load credentials from .env or environment
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL",
                                         "text-embedding-ada-002")

        # Batch embed (ada-002 supports up to 2048 inputs per call)
        all_embeddings = []
        batch_size = 2048
        for i in range(0, len(decoded_tokens), batch_size):
            batch = decoded_tokens[i:i + batch_size]
            response = client.embeddings.create(
                input=batch,
                model=embedding_model
            )
            for item in response.data:
                all_embeddings.append(item.embedding)
            print(f"      Embedded {min(i + batch_size, len(decoded_tokens))}/"
                  f"{len(decoded_tokens)}", flush=True)

        _ada_embeddings = np.array(all_embeddings, dtype=np.float32)

        # Cache to disk
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, token_ids=_ada_numeric_ids,
                 embeddings=_ada_embeddings)
        print(f"    [InferDPT-F] Cached embeddings to {cache_path}", flush=True)

    # Build index mapping
    _ada_id_to_idx = {int(tid): i for i, tid in enumerate(_ada_numeric_ids)}

    # Compute per-dimension sensitivity: range of each dimension
    _ada_per_dim_sens = _ada_embeddings.max(axis=0) - _ada_embeddings.min(axis=0)
    _ada_per_dim_sens = np.maximum(_ada_per_dim_sens, 1e-10)

    print(f"    [InferDPT-F] {len(_ada_numeric_ids)} numeric tokens, "
          f"embedding dim={_ada_embeddings.shape[1]}, "
          f"mean per-dim sensitivity={_ada_per_dim_sens.mean():.6f}",
          flush=True)


def _inferdpt_laplace_noise_embedding(embedding, epsilon):
    """Add Laplace noise to an embedding vector.

    Uses InferDPT's piecewise epsilon formula adapted per dimension:
    sensitivity per dimension = range of that dimension across all
    numeric token embeddings.
    """
    dim = len(embedding)
    noisy = np.copy(embedding)

    for d in range(dim):
        sens = _ada_per_dim_sens[d]
        log_arg = epsilon * 19.0647 - 38.1294
        if epsilon <= 2 or log_arg <= 0:
            beta = sens / epsilon
        else:
            tt = 0.01658 * math.log(log_arg) + 9.3110
            beta = sens / tt
        noisy[d] += np.random.laplace(0, beta)

    return noisy


def _inferdpt_exponential_mechanism_embedding(original_emb, noisy_emb, epsilon):
    """InferDPT's exponential mechanism in embedding space.

    Keeps only candidate tokens whose embedding is closer to the original
    than the noisy embedding, then samples proportionally to
    exp(ε/2 * (d_noise - d_candidate) / d_noise).

    Returns:
        Selected token ID from the filtered numeric vocabulary.
    """
    d_noise = np.linalg.norm(original_emb - noisy_emb)

    if d_noise < 1e-12:
        # Noise is negligible — return nearest numeric token
        dists = np.linalg.norm(_ada_embeddings - original_emb[np.newaxis, :], axis=1)
        return int(_ada_numeric_ids[np.argmin(dists)])

    # Compute distances from original to all numeric token embeddings
    d_candidates = np.linalg.norm(
        _ada_embeddings - original_emb[np.newaxis, :], axis=1)

    # Keep candidates closer to original than the noisy value
    mask = d_candidates < d_noise
    valid_indices = np.where(mask)[0]

    if len(valid_indices) == 0:
        # No candidates within cutoff — return nearest
        return int(_ada_numeric_ids[np.argmin(d_candidates)])

    valid_dists = d_candidates[valid_indices]

    # Exponential mechanism scores
    scores = np.exp((epsilon / 2.0) * (d_noise - valid_dists) / d_noise)
    total = scores.sum()
    if total == 0 or not np.isfinite(total):
        return int(_ada_numeric_ids[valid_indices[np.argmin(valid_dists)]])

    probs = scores / total
    idx = np.random.choice(len(valid_indices), p=probs)
    return int(_ada_numeric_ids[valid_indices[idx]])


def sanitize_inferdpt_filtered(samples, nodes, edges, expressions, epsilon,
                                target_keys, template_name=None):
    """InferDPT-F: embedding-based filtered sanitization.

    Tokenizes each value with RoBERTa, applies InferDPT's two-step mechanism
    (Laplace noise + exponential mechanism) in ada-002 embedding space per
    token, restricted to numeric-only tokens.

    Budget: each node gets uniform epsilon, split across BPE tokens
    (eps/n_tokens per token, basic composition).

    Args:
        samples: List of sample dicts
        nodes: Template node definitions
        edges: Template edge definitions
        expressions: Template expressions (not used, API compatibility)
        epsilon: Per-node privacy budget
        target_keys: Template target key mapping (not used)
        template_name: Template name (for value formatting)

    Returns:
        (all_values, epsilon_blocks)
    """
    _load_ada_embeddings()
    san = Sanitizer()

    epsilon_blocks = []
    all_values = []
    fallback_count = 0
    total_values = 0

    # Get value format order for this template
    value_order = _TEMPLATE_VALUE_ORDER.get(template_name, [])
    node_decimals = {name: dec for name, dec in value_order}

    for si, sample in enumerate(samples):
        topo_order, gt_values = get_topological_order(sample, edges)
        sanitized = {}
        budgets = []

        for node in topo_order:
            if node in _SKIP_NODES:
                sanitized[node] = gt_values[node]
                budgets.append(0)
                continue

            if node not in nodes:
                sanitized[node] = gt_values[node]
                budgets.append(0)
                continue

            total_values += 1
            value = float(gt_values[node])
            lo, hi = MEDICAL_DOMAINS.get(node, MEDICAL_DOMAINS["_default"])

            # Format value as string
            decimals = node_decimals.get(node, 2)
            if decimals == 0:
                val_str = str(int(value))
            else:
                val_str = f"{value:.{decimals}f}"

            # Tokenize with RoBERTa
            token_ids = _roberta_tokenizer.encode(val_str,
                                                   add_special_tokens=False)

            # Structural tokens ("." decimal, "-" sign) are preserved.
            # Only digit tokens consume privacy budget.
            sanitizable = [(i, tid) for i, tid in enumerate(token_ids)
                           if _roberta_tokenizer.decode([tid]).strip() not in ('.', '-')]
            n_sanitizable = len(sanitizable)
            eps_per_tok = epsilon / max(n_sanitizable, 1)

            # Replace each sanitizable token via InferDPT in embedding space
            replacement_ids = list(token_ids)  # start with original
            all_in_vocab = True
            for pos, tid in sanitizable:
                if tid not in _ada_id_to_idx:
                    all_in_vocab = False
                    break
                idx = _ada_id_to_idx[tid]
                orig_emb = _ada_embeddings[idx]
                noisy_emb = _inferdpt_laplace_noise_embedding(orig_emb,
                                                               eps_per_tok)
                new_tid = _inferdpt_exponential_mechanism_embedding(
                    orig_emb, noisy_emb, eps_per_tok)
                replacement_ids[pos] = new_tid

            if all_in_vocab:
                replacement_str = _roberta_tokenizer.decode(
                    replacement_ids, skip_special_tokens=True).strip()
                match = re.search(r'-?\d+\.?\d*', replacement_str)
                if match:
                    san_val = float(match.group())
                    san_val = np.clip(san_val, lo, hi)
                    sanitized[node] = round(float(san_val), ndigits=4)
                else:
                    fallback_count += 1
                    noised = san.M_exponential_discrete(value, epsilon, lo, hi)
                    sanitized[node] = round(float(noised), ndigits=4)
            else:
                fallback_count += 1
                noised = san.M_exponential_discrete(value, epsilon, lo, hi)
                sanitized[node] = round(float(noised), ndigits=4)

            budgets.append(epsilon)

        all_values.append(sanitized)
        epsilon_blocks.append(budgets)

        if (si + 1) % 100 == 0:
            print(f"      [InferDPT-F] {si+1}/{len(samples)} samples done",
                  flush=True)

    if total_values > 0:
        rate = fallback_count / total_values * 100
        print(f"    [InferDPT-F] Fallback rate: {fallback_count}/{total_values} "
              f"({rate:.1f}%)", flush=True)

    return all_values, epsilon_blocks
