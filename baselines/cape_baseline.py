"""
CAPE Baseline — vanilla text adaptation (ICML 2025)

Reference: Wu et al., "CAPE: Context-Aware Prompt Perturbation Mechanism
with Differential Privacy", ICML 2025.

Algorithm per sensitive token at position i:
  1. Mask the token and run a masked LM (BERT) to get logits L_r for
     every vocabulary token r.
  2. Clip logits to [-C, C] to bound the utility sensitivity.
  3. Compute static embedding distances D(t_i, r) for all vocab tokens.
  4. Compute hybrid utility:  u(r) = clip(L_r) · sim(t_i, r)
     where sim = 1 - D_norm (normalized to [0,1]).
  5. Partition vocabulary into N_b equal-width buckets by utility score.
  6. Compute aggregate bucket utility μ_j = Σ u(r) for r in bucket j.
  7. Sample bucket j via exponential mechanism:
     Pr[j] ∝ exp(ε · μ_j / (2·Δ))
  8. Sample uniformly within the selected bucket.

Only numeric tokens are perturbed. Total budget = n_nodes × ε,
distributed uniformly across numeric tokens.

Optimization: all numeric token positions within a sample are batched
into a single MLM forward pass (one masked copy per position).

Privacy guarantee: (ε + ε')-DP per token where
  ε' = ln(max|b_j| / min|b_j|) for bucket size imbalance.
Under equal-size buckets ε' ≈ 0, giving pure ε-DP.
Basic composition across sanitized tokens.
"""

import re
import numpy as np

from preempt.sanitizer import MEDICAL_DOMAINS, Sanitizer
from utils.utils import get_topological_order
from baselines.dp_gtr_baseline import sample_to_paragraph

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

_NUMBER_PATTERN = re.compile(r'-?\d+\.?\d*')

# Hyperparameters (from paper: N_b=50, clipping threshold tunable)
_NUM_BUCKETS = 50
_LOGIT_CLIP = 10.0    # clip logits to [-C, C]
_LAMBDA_L = 0.5       # logit weight exponent (paper: λ_L = 0.5)
_LAMBDA_D = 1.0       # distance weight exponent

# Lazy-loaded models
_mlm_model = None
_tokenizer = None
_static_embs = None
_vocab_size = None
_dist_cache = {}  # token_id → (dists, similarity) cached per unique token
_numeric_vocab_mask = None  # boolean mask: True for tokens that decode to numeric chars


def _load_models():
    """Lazy-load BERT MLM model, tokenizer, and static embeddings."""
    global _mlm_model, _tokenizer, _static_embs, _vocab_size

    if _mlm_model is not None:
        return

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model_name = "bert-base-uncased"
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)
    _mlm_model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _mlm_model = _mlm_model.to(device)

    # Extract static word embeddings for distance computation
    _static_embs = (_mlm_model.bert.embeddings.word_embeddings
                    .weight.detach().cpu().numpy())
    _vocab_size = _static_embs.shape[0]

    # Build numeric vocabulary mask (must contain at least one digit)
    global _numeric_vocab_mask
    numeric_re = re.compile(r'^[\d.\-]*\d[\d.\-]*$')
    _numeric_vocab_mask = np.zeros(_vocab_size, dtype=bool)
    for idx in range(_vocab_size):
        decoded = _tokenizer.decode([idx]).strip()
        # BERT WordPiece: strip ## prefix from continuation tokens
        if decoded.startswith('##'):
            decoded = decoded[2:]
        if decoded and numeric_re.match(decoded):
            _numeric_vocab_mask[idx] = True
    n_numeric_tokens = _numeric_vocab_mask.sum()

    print(f"    [CAPE] Loaded BERT MLM on {device}, vocab_size={_vocab_size}, "
          f"numeric_tokens={n_numeric_tokens}", flush=True)


def _get_mlm_logits_batched(token_ids, mask_positions):
    """
    Get MLM logits at multiple masked positions in a single batched forward pass.

    Creates one copy of the sequence per mask position, each with a different
    token masked, and runs them all through the model at once.

    Args:
        token_ids: List of token IDs (without BOS/EOS)
        mask_positions: List of int positions to mask

    Returns:
        numpy array of shape (len(mask_positions), vocab_size) — logits per position
    """
    import torch

    _load_models()
    device = next(_mlm_model.parameters()).device

    cls_id = _tokenizer.cls_token_id
    sep_id = _tokenizer.sep_token_id
    mask_id = _tokenizer.mask_token_id

    # Build batch: one sequence per mask position
    batch = []
    for pos in mask_positions:
        ids = list(token_ids)
        ids[pos] = mask_id
        batch.append([cls_id] + ids + [sep_id])

    input_tensor = torch.tensor(batch, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = _mlm_model(input_tensor)
        # outputs.logits shape: (batch_size, seq_len, vocab_size)
        # Extract logits at each masked position (+1 for BOS offset)
        all_logits = []
        for i, pos in enumerate(mask_positions):
            all_logits.append(outputs.logits[i, pos + 1, :].cpu().numpy())

    return np.array(all_logits)  # (n_positions, vocab_size)


def _cape_select(logits, target_id, epsilon):
    """
    CAPE bucketized selection for one token given pre-computed MLM logits.

    Fully vectorized — no Python loops over the 50K vocabulary.

    Returns:
        Replacement token ID
    """
    # Step 1-2: Clip logits, compute similarity (cached per unique token)
    clipped = np.clip(logits, -_LOGIT_CLIP, _LOGIT_CLIP)
    shifted = clipped + _LOGIT_CLIP  # ∈ [0, 2C]

    if target_id in _dist_cache:
        similarity = _dist_cache[target_id]
    else:
        static_emb = _static_embs[target_id]
        dists = np.linalg.norm(_static_embs - static_emb[np.newaxis, :], axis=1)
        d_min = dists.min()
        d_max = dists.max()
        d_norm = (dists - d_min) / (d_max - d_min + 1e-10)
        similarity = np.exp(-d_norm)  # paper Eq. 4-6: D = exp(-D_norm)
        _dist_cache[target_id] = similarity

    # Step 3: Hybrid utility (vectorized)
    utility = (shifted ** _LAMBDA_L) * (similarity ** _LAMBDA_D)

    # Step 4: Bucketize (vectorized)
    u_min, u_max = utility.min(), utility.max()
    bucket_width = (u_max - u_min + 1e-10) / _NUM_BUCKETS
    bucket_ids = np.clip(
        ((utility - u_min) / bucket_width).astype(int),
        0, _NUM_BUCKETS - 1
    )

    # Bucket representative score = MEAN utility (paper Algorithm 1)
    bucket_sums = np.bincount(bucket_ids, weights=utility,
                              minlength=_NUM_BUCKETS)
    bucket_sizes = np.bincount(bucket_ids, minlength=_NUM_BUCKETS)
    bucket_means = np.where(bucket_sizes > 0, bucket_sums / bucket_sizes, 0.0)

    # Step 5: Exponential mechanism over bucket means
    # Paper Theorem 4.1: guarantee is (ε + ε')-DP where ε' = ln(max|b|/min|b|).
    # Paper Algorithm 2 uses ε directly (no subtraction).
    nonempty = bucket_sizes > 0

    # Sensitivity: utility ∈ [0, (2C)^λ_L], so Δ = (2C)^λ_L
    delta = (2.0 * _LOGIT_CLIP) ** _LAMBDA_L
    log_probs = epsilon * bucket_means / (2.0 * delta)
    log_probs[~nonempty] = -np.inf
    if nonempty.any():
        log_probs -= np.max(log_probs[nonempty])
    probs = np.exp(log_probs)
    total = probs.sum()
    if total <= 0:
        probs = nonempty.astype(float)
        total = probs.sum()
    probs /= total

    selected_bucket = np.random.choice(_NUM_BUCKETS, p=probs)

    # Select uniformly from the chosen bucket (vectorized mask)
    members = np.where(bucket_ids == selected_bucket)[0]
    return int(np.random.choice(members))


def _cape_select_filtered(logits, target_id, epsilon):
    """
    CAPE bucketized selection restricted to numeric-only vocabulary.

    Same mechanism as _cape_select but zeroes out utility for all non-numeric
    tokens before bucketizing. This guarantees the output is always a numeric
    token (digit, decimal point, minus sign), eliminating the need for fallback.

    The privacy guarantee still holds: restricting the output space is equivalent
    to running the exponential mechanism over a smaller candidate set.
    """
    clipped = np.clip(logits, -_LOGIT_CLIP, _LOGIT_CLIP)
    shifted = clipped + _LOGIT_CLIP

    if target_id in _dist_cache:
        similarity = _dist_cache[target_id]
    else:
        static_emb = _static_embs[target_id]
        dists = np.linalg.norm(_static_embs - static_emb[np.newaxis, :], axis=1)
        d_min = dists.min()
        d_max = dists.max()
        d_norm = (dists - d_min) / (d_max - d_min + 1e-10)
        similarity = np.exp(-d_norm)  # paper Eq. 4-6: D = exp(-D_norm)
        _dist_cache[target_id] = similarity

    utility = (shifted ** _LAMBDA_L) * (similarity ** _LAMBDA_D)

    # Zero out non-numeric tokens — they cannot be selected
    utility[~_numeric_vocab_mask] = 0.0

    # Only bucketize over numeric tokens
    numeric_ids = np.where(_numeric_vocab_mask)[0]
    numeric_utility = utility[numeric_ids]

    u_min, u_max = numeric_utility.min(), numeric_utility.max()
    bucket_width = (u_max - u_min + 1e-10) / _NUM_BUCKETS
    bucket_ids = np.clip(
        ((numeric_utility - u_min) / bucket_width).astype(int),
        0, _NUM_BUCKETS - 1
    )

    bucket_sums = np.bincount(bucket_ids, weights=numeric_utility,
                              minlength=_NUM_BUCKETS)
    bucket_sizes = np.bincount(bucket_ids, minlength=_NUM_BUCKETS)
    bucket_means = np.where(bucket_sizes > 0, bucket_sums / bucket_sizes, 0.0)

    nonempty = bucket_sizes > 0

    delta = (2.0 * _LOGIT_CLIP) ** _LAMBDA_L
    log_probs = epsilon * bucket_means / (2.0 * delta)
    log_probs[~nonempty] = -np.inf
    if nonempty.any():
        log_probs -= np.max(log_probs[nonempty])
    probs = np.exp(log_probs)
    total = probs.sum()
    if total <= 0:
        probs = nonempty.astype(float)
        total = probs.sum()
    probs /= total

    selected_bucket = np.random.choice(_NUM_BUCKETS, p=probs)
    members = numeric_ids[bucket_ids == selected_bucket]
    return int(np.random.choice(members))


def _find_value_token_spans(token_ids, values, value_order):
    """Map each node's formatted value to its BPE token positions in the paragraph.

    Strategy: decode the full token sequence into a character string, keeping
    track of which character positions map to which token indices. Then for
    each node value, find its formatted string in the decoded text and map
    the character span back to token indices.

    This avoids BPE encoding mismatches (leading space, merges) by working
    entirely in decoded text space.

    Returns:
        List of (node_name, start_token, end_token) for each value.
        Token positions are indices into token_ids. (-1, -1) if not found.
    """
    # Build character-to-token mapping by decoding one token at a time
    char_to_token = []  # char_to_token[char_idx] = token_idx
    decoded_parts = []
    for tidx, tid in enumerate(token_ids):
        part = _tokenizer.decode([tid])
        for _ in part:
            char_to_token.append(tidx)
        decoded_parts.append(part)
    full_text = "".join(decoded_parts)

    spans = []
    search_from_char = 0

    for node_name, decimals in value_order:
        if node_name not in values:
            continue
        val = values[node_name]
        if decimals == 0:
            val_str = str(int(val))
        else:
            val_str = f"{float(val):.{decimals}f}"

        # Search for the value string in the decoded text
        char_pos = full_text.find(val_str, search_from_char)
        if char_pos >= 0:
            char_end = char_pos + len(val_str)
            # Map character span to token span
            start_token = char_to_token[char_pos]
            end_token = char_to_token[min(char_end - 1, len(char_to_token) - 1)] + 1
            spans.append((node_name, start_token, end_token))
            search_from_char = char_end
        else:
            spans.append((node_name, -1, -1))

    return spans


def sanitize_cape(samples, nodes, edges, epsilon, template_target_keys,
                  template_name=None):
    """
    Sanitize medical values via CAPE context-aware bucketized DP.

    For each node: generate paragraph for context → find the node's value
    tokens in the BPE sequence → get MLM logits at those positions (batched)
    → apply CAPE bucketized selection per token → decode replacement tokens
    → parse as float → assign directly to the correct node.

    Budget: each node gets uniform epsilon. Per-token budget within a
    multi-token value is epsilon / n_tokens (basic composition).
    """
    _load_models()
    san = Sanitizer()
    epsilon_blocks = []
    all_values = []
    fallback_count = 0
    total_values = 0

    value_order = _TEMPLATE_VALUE_ORDER.get(template_name, [])

    for si, sample in enumerate(samples):
        topo_order, gt_values = get_topological_order(sample, edges)
        nodes_to_release = [n for n in topo_order if n not in _SKIP_NODES]

        # Step 1: Generate paragraph for context, tokenize
        paragraph = sample_to_paragraph(sample, template_name)
        token_ids = _tokenizer.encode(paragraph, add_special_tokens=False)

        # Step 2: Map each node's value to its BPE token span in the paragraph
        spans = _find_value_token_spans(token_ids, gt_values, value_order)

        # Collect all token positions that need masking (across all nodes)
        all_mask_positions = []
        span_to_mask_indices = {}  # node_name → list of indices into all_mask_positions
        for node_name, start, end in spans:
            if start < 0:
                continue
            indices = []
            for pos in range(start, end):
                indices.append(len(all_mask_positions))
                all_mask_positions.append(pos)
            span_to_mask_indices[node_name] = indices

        # Step 3: Single batched MLM forward pass for all positions
        all_logits = None
        if all_mask_positions:
            all_logits = _get_mlm_logits_batched(token_ids, all_mask_positions)

        # Step 4: Per-node — apply CAPE selection, decode, parse
        sanitized = {}
        budgets = []
        for node in nodes_to_release:
            total_values += 1
            domain_lower, domain_upper = MEDICAL_DOMAINS.get(
                node, MEDICAL_DOMAINS["_default"])
            true_val = float(gt_values[node])

            if node in span_to_mask_indices and all_logits is not None:
                mask_indices = span_to_mask_indices[node]
                n_tokens = len(mask_indices)
                # Budget per BPE token: epsilon / n_tokens (basic composition)
                eps_per_tok = epsilon / max(n_tokens, 1)

                # Apply CAPE selection to each token of this value
                replacement_ids = []
                for midx in mask_indices:
                    orig_tid = token_ids[all_mask_positions[midx]]
                    new_id = _cape_select(all_logits[midx], orig_tid, eps_per_tok)
                    replacement_ids.append(new_id)

                # Decode replacement tokens to string
                replacement_str = _tokenizer.decode(replacement_ids,
                                                    skip_special_tokens=True).strip()
                # Parse as float
                try:
                    san_val = float(replacement_str)
                    san_val = np.clip(san_val, domain_lower, domain_upper)
                    sanitized[node] = round(float(san_val), ndigits=4)
                except ValueError:
                    # Non-numeric replacement → fallback
                    fallback_count += 1
                    noised = san.M_exponential_discrete(
                        true_val, epsilon, domain_lower, domain_upper)
                    sanitized[node] = round(float(noised), ndigits=4)
            else:
                # Value not found in paragraph → fallback
                fallback_count += 1
                noised = san.M_exponential_discrete(
                    true_val, epsilon, domain_lower, domain_upper)
                sanitized[node] = round(float(noised), ndigits=4)

            budgets.append(epsilon)

        all_values.append(sanitized)
        epsilon_blocks.append(budgets)

        if (si + 1) % 100 == 0:
            print(f"      [CAPE] {si+1}/{len(samples)} samples done", flush=True)

    if total_values > 0:
        rate = fallback_count / total_values * 100
        print(f"    [CAPE] Fallback rate: {fallback_count}/{total_values} "
              f"({rate:.1f}%)", flush=True)

    return all_values, epsilon_blocks


def sanitize_cape_filtered(samples, nodes, edges, epsilon, template_target_keys,
                           template_name=None):
    """
    CAPE with numeric-filtered vocabulary (CAPE-F).

    Same as sanitize_cape but uses _cape_select_filtered: the exponential
    mechanism only considers tokens that decode to numeric characters
    (digits, decimal points, minus signs). This guarantees every BPE
    replacement is numeric, eliminating fallback entirely.

    Tests whether CAPE's context-aware MLM logits help when the output
    space is restricted to numbers.
    """
    _load_models()
    san = Sanitizer()
    epsilon_blocks = []
    all_values = []
    fallback_count = 0
    total_values = 0

    value_order = _TEMPLATE_VALUE_ORDER.get(template_name, [])

    for si, sample in enumerate(samples):
        topo_order, gt_values = get_topological_order(sample, edges)
        nodes_to_release = [n for n in topo_order if n not in _SKIP_NODES]

        paragraph = sample_to_paragraph(sample, template_name)
        token_ids = _tokenizer.encode(paragraph, add_special_tokens=False)

        spans = _find_value_token_spans(token_ids, gt_values, value_order)

        all_mask_positions = []
        span_to_mask_indices = {}
        for node_name, start, end in spans:
            if start < 0:
                continue
            indices = []
            for pos in range(start, end):
                indices.append(len(all_mask_positions))
                all_mask_positions.append(pos)
            span_to_mask_indices[node_name] = indices

        all_logits = None
        if all_mask_positions:
            all_logits = _get_mlm_logits_batched(token_ids, all_mask_positions)

        sanitized = {}
        budgets = []
        for node in nodes_to_release:
            total_values += 1
            domain_lower, domain_upper = MEDICAL_DOMAINS.get(
                node, MEDICAL_DOMAINS["_default"])
            true_val = float(gt_values[node])

            if node in span_to_mask_indices and all_logits is not None:
                mask_indices = span_to_mask_indices[node]
                n_tokens = len(mask_indices)
                eps_per_tok = epsilon / max(n_tokens, 1)

                replacement_ids = []
                for midx in mask_indices:
                    orig_tid = token_ids[all_mask_positions[midx]]
                    # Use filtered selector — guaranteed numeric output
                    new_id = _cape_select_filtered(all_logits[midx], orig_tid,
                                                   eps_per_tok)
                    replacement_ids.append(new_id)

                replacement_str = _tokenizer.decode(replacement_ids,
                                                    skip_special_tokens=True).strip()
                match = _NUMBER_PATTERN.search(replacement_str)
                if match:
                    san_val = float(match.group())
                    san_val = np.clip(san_val, domain_lower, domain_upper)
                    sanitized[node] = round(float(san_val), ndigits=4)
                else:
                    fallback_count += 1
                    noised = san.M_exponential_discrete(
                        true_val, epsilon, domain_lower, domain_upper)
                    sanitized[node] = round(float(noised), ndigits=4)
            else:
                fallback_count += 1
                noised = san.M_exponential_discrete(
                    true_val, epsilon, domain_lower, domain_upper)
                sanitized[node] = round(float(noised), ndigits=4)

            budgets.append(epsilon)

        all_values.append(sanitized)
        epsilon_blocks.append(budgets)

        if (si + 1) % 100 == 0:
            print(f"      [CAPE-F] {si+1}/{len(samples)} samples done",
                  flush=True)

    if total_values > 0:
        rate = fallback_count / total_values * 100
        print(f"    [CAPE-F] Fallback rate: {fallback_count}/{total_values} "
              f"({rate:.1f}%)", flush=True)

    return all_values, epsilon_blocks


def sanitize_cape_filtered_roots(samples, nodes, edges, expressions, epsilon,
                                 template_target_keys, template_name=None,
                                 num_candidates=1000):
    """
    CAPE-F Roots: noise only root nodes via CAPE-F, post-process derived nodes.

    Total budget B = n_nodes * ε, split uniformly across k roots.
    Each root's share is further split across its BPE tokens.
    """
    import math
    import random as random_mod

    _load_models()
    san = Sanitizer()
    epsilon_blocks = []
    all_values = []
    fallback_count = 0
    total_values = 0

    value_order = _TEMPLATE_VALUE_ORDER.get(template_name, [])
    skip_nodes = list(_SKIP_NODES)

    for si, sample in enumerate(samples):
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

        n_nodes = len([n for n in topo_order if n not in skip_nodes])
        total_budget = n_nodes * epsilon
        k = len(root_nodes)
        root_eps = total_budget / k if k > 0 else epsilon

        # Generate paragraph and tokenize for MLM context
        paragraph = sample_to_paragraph(sample, template_name)
        token_ids = _tokenizer.encode(paragraph, add_special_tokens=False)
        spans = _find_value_token_spans(token_ids, gt_values, value_order)

        # Collect mask positions for ROOT nodes only
        all_mask_positions = []
        span_to_mask_indices = {}
        for node_name, start, end in spans:
            if start < 0 or node_name not in root_nodes:
                continue
            indices = []
            for pos in range(start, end):
                indices.append(len(all_mask_positions))
                all_mask_positions.append(pos)
            span_to_mask_indices[node_name] = indices

        all_logits = None
        if all_mask_positions:
            all_logits = _get_mlm_logits_batched(token_ids, all_mask_positions)

        # Noise root nodes via CAPE-F
        for root in root_nodes:
            total_values += 1
            lo, hi = MEDICAL_DOMAINS.get(root, MEDICAL_DOMAINS["_default"])
            true_val = float(ground_truth[root])

            if root in span_to_mask_indices and all_logits is not None:
                mask_indices = span_to_mask_indices[root]
                n_tokens = len(mask_indices)
                eps_per_tok = root_eps / max(n_tokens, 1)

                replacement_ids = []
                for midx in mask_indices:
                    orig_tid = token_ids[all_mask_positions[midx]]
                    new_id = _cape_select_filtered(all_logits[midx], orig_tid,
                                                   eps_per_tok)
                    replacement_ids.append(new_id)

                replacement_str = _tokenizer.decode(replacement_ids,
                                                    skip_special_tokens=True).strip()
                match = _NUMBER_PATTERN.search(replacement_str)
                if match:
                    san_val = float(match.group())
                    san_val = np.clip(san_val, lo, hi)
                    sanitized_values[root] = round(float(san_val), ndigits=4)
                else:
                    fallback_count += 1
                    noised = san.M_exponential_discrete(true_val, root_eps, lo, hi)
                    sanitized_values[root] = round(float(noised), ndigits=4)
            else:
                fallback_count += 1
                noised = san.M_exponential_discrete(true_val, root_eps, lo, hi)
                sanitized_values[root] = round(float(noised), ndigits=4)

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
                    {"random": random_mod, "round": round, "np": np,
                     "math": math},
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

        if (si + 1) % 100 == 0:
            print(f"      [CAPE-F-R] {si+1}/{len(samples)} samples done",
                  flush=True)

    if total_values > 0:
        rate = fallback_count / total_values * 100
        print(f"    [CAPE-F-R] Fallback rate: {fallback_count}/{total_values} "
              f"({rate:.1f}%)", flush=True)

    return all_values, epsilon_blocks


def sanitize_cape_filtered_roots_opt(samples, nodes, edges, expressions, epsilon,
                                     template_target_keys, population_means,
                                     template_name=None, num_candidates=1000,
                                     epsilon_min=0.005):
    """
    CAPE-F Roots with P++'s optimal budget allocation.

    Same as sanitize_cape_filtered_roots but replaces uniform B/k allocation
    with sensitivity-weighted allocation from optimal_budget_allocation_exact.
    """
    import math
    import random as random_mod
    from utils.utils import (optimal_budget_allocation_exact,
                             compute_population_mean_sensitivities)
    from preempt import sanitizer

    _load_models()
    san = Sanitizer()
    epsilon_blocks = []
    all_values = []
    fallback_count = 0
    total_values = 0
    _sens_cache = {}

    value_order = _TEMPLATE_VALUE_ORDER.get(template_name, [])
    skip_nodes = list(_SKIP_NODES)

    for si, sample in enumerate(samples):
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
        for tname, tnode in template_target_keys.items():
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

        # Generate paragraph and tokenize for MLM context
        paragraph = sample_to_paragraph(sample, template_name)
        token_ids = _tokenizer.encode(paragraph, add_special_tokens=False)
        spans = _find_value_token_spans(token_ids, gt_values, value_order)

        # Collect mask positions for ROOT nodes only
        all_mask_positions = []
        span_to_mask_indices = {}
        for node_name, start, end in spans:
            if start < 0 or node_name not in root_nodes:
                continue
            indices = []
            for pos in range(start, end):
                indices.append(len(all_mask_positions))
                all_mask_positions.append(pos)
            span_to_mask_indices[node_name] = indices

        all_logits = None
        if all_mask_positions:
            all_logits = _get_mlm_logits_batched(token_ids, all_mask_positions)

        # Noise root nodes via CAPE-F with optimal allocation
        for root in root_nodes:
            total_values += 1
            root_eps = allocation.get(root, epsilon)
            lo, hi = MEDICAL_DOMAINS.get(root, MEDICAL_DOMAINS["_default"])
            true_val = float(ground_truth[root])

            if root in span_to_mask_indices and all_logits is not None:
                mask_indices = span_to_mask_indices[root]
                n_tokens = len(mask_indices)
                eps_per_tok = root_eps / max(n_tokens, 1)

                replacement_ids = []
                for midx in mask_indices:
                    orig_tid = token_ids[all_mask_positions[midx]]
                    new_id = _cape_select_filtered(all_logits[midx], orig_tid,
                                                   eps_per_tok)
                    replacement_ids.append(new_id)

                replacement_str = _tokenizer.decode(replacement_ids,
                                                    skip_special_tokens=True).strip()
                match = _NUMBER_PATTERN.search(replacement_str)
                if match:
                    san_val = float(match.group())
                    san_val = np.clip(san_val, lo, hi)
                    sanitized_values[root] = round(float(san_val), ndigits=4)
                else:
                    fallback_count += 1
                    noised = san.M_exponential_discrete(true_val, root_eps, lo, hi)
                    sanitized_values[root] = round(float(noised), ndigits=4)
            else:
                fallback_count += 1
                noised = san.M_exponential_discrete(true_val, root_eps, lo, hi)
                sanitized_values[root] = round(float(noised), ndigits=4)

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
                    {"random": random_mod, "round": round, "np": np,
                     "math": math},
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

        if (si + 1) % 100 == 0:
            print(f"      [CAPE-F-RO] {si+1}/{len(samples)} samples done",
                  flush=True)

    if total_values > 0:
        rate = fallback_count / total_values * 100
        print(f"    [CAPE-F-RO] Fallback rate: {fallback_count}/{total_values} "
              f"({rate:.1f}%)", flush=True)

    return all_values, epsilon_blocks
