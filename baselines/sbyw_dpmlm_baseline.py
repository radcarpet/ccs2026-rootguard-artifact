"""
SBYW + DP-MLM Baseline

Combines NLP-based privacy budget allocation (SBYW / EpsilonDistributor) with
DP-MLM (exponential mechanism over RoBERTa MLM logits) for numeric value
sanitization.

Pipeline per sample:
  1. Generate paragraph from ground truth using DP-GTR templates
  2. Run SBYW importance scoring on the paragraph
  3. Extract importance weights for numeric value tokens only
  4. Redistribute total budget (n_nodes * epsilon) across numeric values,
     weighted by SBYW importance (inverse: important → less epsilon)
  5. Replace each numeric value via DP-MLM at BPE subtoken level
  6. Parse replacement as float
  7. Handle non-numeric output:
       Version A (filter_numeric=False): use as-is if numeric, fallback to
           M_exponential if non-numeric
       Version B (filter_numeric=True): mask non-numeric BPE tokens before
           sampling so output is always numeric, clip to domain bounds

Privacy accounting:
  - SBYW is a scoring mechanism — no privacy cost
  - Paragraph is generated from ground truth — not released
  - Each value gets epsilon from renormalized SBYW distribution
  - BPE subtokens with eps/k each compose to eps per word (basic composition)
  - Total = sum(node_epsilons) = n_nodes * epsilon (matching vanilla)
"""

import re
import numpy as np
import torch
import nltk

from preempt.sanitizer import MEDICAL_DOMAINS, Sanitizer
from utils.utils import get_topological_order
from baselines.dp_gtr_baseline import sample_to_paragraph, _fmt

# ---------------------------------------------------------------------------
# Template value ordering — maps each template to the ordered list of
# (node_name, decimal_places) as they appear in the paragraph.
# ---------------------------------------------------------------------------

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

# Nodes that are classification outputs — skip sanitization
_SKIP_NODES = frozenset([
    "anemia_class", "fib4_risk", "aip_risk", "ci_risk",
    "ppi_risk", "tyg_class", "homa_class", "nlr_class",
])

# ---------------------------------------------------------------------------
# Lazy-loaded singleton models
# ---------------------------------------------------------------------------

_cached_eps_distributor = None
_cached_dpmlm = None
_cached_sanitizer = None
_cached_numeric_vocab_mask = None


def _get_eps_distributor():
    global _cached_eps_distributor
    if _cached_eps_distributor is None:
        from baselines.EpsilonDistributor.EpsilonDistributor import EpsilonDistributor
        _cached_eps_distributor = EpsilonDistributor()
    return _cached_eps_distributor


def _get_dpmlm():
    global _cached_dpmlm
    if _cached_dpmlm is None:
        from baselines.DPMLM.DPMLM import DPMLM
        _cached_dpmlm = DPMLM()
    return _cached_dpmlm


def _get_sanitizer():
    global _cached_sanitizer
    if _cached_sanitizer is None:
        _cached_sanitizer = Sanitizer()
    return _cached_sanitizer


# ---------------------------------------------------------------------------
# Numeric vocabulary mask (for Version B)
# ---------------------------------------------------------------------------

def _build_numeric_vocab_mask(tokenizer):
    """Build a boolean mask over the full vocabulary: True for numeric tokens.

    A token is considered numeric if its decoded string (stripped) matches
    digits, decimal points, minus signs, or combinations thereof.
    Uses len(tokenizer) to include any added special tokens beyond base vocab.
    """
    vocab_size = len(tokenizer)
    mask = np.zeros(vocab_size, dtype=bool)
    numeric_pattern = re.compile(r'^[\d.\-]+$')
    for idx in range(vocab_size):
        decoded = tokenizer.decode([idx]).strip()
        if decoded and numeric_pattern.match(decoded):
            mask[idx] = True
    return mask


def _get_numeric_vocab_mask(tokenizer):
    global _cached_numeric_vocab_mask
    if _cached_numeric_vocab_mask is None:
        _cached_numeric_vocab_mask = _build_numeric_vocab_mask(tokenizer)
    return _cached_numeric_vocab_mask


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _build_value_to_node_map(gt_values, template_name):
    """Return ordered list of (formatted_value_string, node_name) matching
    the paragraph order for this template.

    Uses _TEMPLATE_VALUE_ORDER and _fmt to produce the exact string that
    appears in the paragraph.
    """
    order = _TEMPLATE_VALUE_ORDER.get(template_name)
    if order is None:
        return []
    pairs = []
    for node_name, decimals in order:
        if node_name not in gt_values:
            continue
        val = gt_values[node_name]
        if decimals == 0:
            formatted = str(int(val))
        else:
            formatted = _fmt(val, decimals)
        pairs.append((formatted, node_name))
    return pairs


def _extract_value_weights(sbyw_result, value_node_pairs):
    """Extract SBYW importance weights for each numeric value node.

    SBYW assigns None to numeric tokens, so we use the nearest preceding
    label token's weight as a proxy for each value's importance. For values
    that fail to match, we assign the mean of successfully extracted weights
    so they get a fair budget share.

    Args:
        sbyw_result: List of (token, epsilon_or_None) from EpsilonDistributor.
        value_node_pairs: Ordered list of (formatted_value_string, node_name).

    Returns:
        Dict {node_name: sbyw_weight} for matched numeric values.
    """
    weights = {}
    pair_idx = 0
    if not value_node_pairs:
        return weights

    # Track the last non-None weight (label token preceding each value)
    last_label_weight = None

    for token, eps_val in sbyw_result:
        # Update label weight tracker for non-None tokens
        if eps_val is not None and eps_val > 0:
            last_label_weight = eps_val

        if pair_idx >= len(value_node_pairs):
            break
        target_str, node_name = value_node_pairs[pair_idx]

        # Check if this token matches (or is part of) the expected value
        if token == target_str or token in target_str.split('.'):
            # Use the preceding label token's weight
            if last_label_weight is not None:
                w = last_label_weight
            else:
                w = 1.0  # no label seen yet, neutral fallback
            if node_name not in weights:
                weights[node_name] = w
            # Advance to next value once full string matched
            if token == target_str:
                pair_idx += 1
        elif target_str.startswith(token):
            # Partial match (e.g., integer part of "12.34")
            if node_name not in weights:
                w = last_label_weight if last_label_weight is not None else 1.0
                weights[node_name] = w

    # Assign misses the mean of extracted weights (fair share)
    matched_weights = list(weights.values())
    miss_weight = float(np.mean(matched_weights)) if matched_weights else 1.0
    for _, node_name in value_node_pairs:
        if node_name not in weights:
            weights[node_name] = miss_weight

    return weights


def _distribute_budget(value_weights, total_epsilon, epsilon_min=0.005):
    """Redistribute total_epsilon across values proportional to SBYW weights.

    SBYW gives higher epsilon to *less important* tokens (inverse relationship).
    So higher SBYW weight → less important → gets more epsilon here too.
    This preserves the SBYW semantics: important values are protected more.

    Args:
        value_weights: Dict {node_name: sbyw_weight}
        total_epsilon: Total budget to distribute
        epsilon_min: Floor to avoid degenerate zero-budget nodes

    Returns:
        Dict {node_name: allocated_epsilon}
    """
    if not value_weights:
        return {}

    nodes = list(value_weights.keys())
    raw_weights = np.array([value_weights[n] for n in nodes], dtype=float)

    # Normalize weights to proportions
    total_weight = raw_weights.sum()
    if total_weight <= 0:
        # Uniform fallback
        per_node = total_epsilon / len(nodes)
        return {n: max(per_node, epsilon_min) for n in nodes}

    proportions = raw_weights / total_weight
    allocations = proportions * total_epsilon

    # Clamp at epsilon_min and redistribute excess
    clamped = np.maximum(allocations, epsilon_min)
    excess = clamped.sum() - total_epsilon
    if excess > 0:
        # Scale down unclamped nodes proportionally
        above_min = clamped > epsilon_min
        if above_min.any():
            scale = (clamped[above_min].sum() - excess) / clamped[above_min].sum()
            clamped[above_min] *= scale

    return {n: float(clamped[i]) for i, n in enumerate(nodes)}


def _prepare_dpmlm_tasks(dpmlm, paragraph, value_str, epsilon):
    """Prepare batched MLM tasks for a single value in a paragraph.

    Returns a list of task dicts, each containing the info needed for one
    masked-subtoken prediction. These are collected across all values/samples
    and run through the model in a single batched forward pass.

    Returns:
        (tasks, fallback) — list of task dicts, or ([], fallback_result) if
        the BPE subsequence wasn't found.
    """
    tokenizer = dpmlm.tokenizer
    value_token_ids = tokenizer.encode(" " + value_str, add_special_tokens=False)
    k = len(value_token_ids)
    if k == 0:
        return [], value_str

    eps_per_subtoken = epsilon / k
    full_ids = tokenizer.encode(paragraph, add_special_tokens=False)
    subseq_start = _find_subsequence(full_ids, value_token_ids)

    if subseq_start is None:
        # Can't find value in BPE — will need word-level fallback
        return [], None

    tasks = []
    for sub_idx, sub_token_id in enumerate(value_token_ids):
        masked_ids = full_ids.copy()
        masked_ids[subseq_start + sub_idx] = tokenizer.mask_token_id
        masked_sent = tokenizer.decode(masked_ids, skip_special_tokens=False)

        input_ids = tokenizer.encode(
            " " + paragraph.replace("MASK", ""),
            " " + masked_sent,
            add_special_tokens=True,
            truncation="only_first"
        )

        try:
            mask_pos = input_ids.index(tokenizer.mask_token_id)
        except ValueError:
            mask_pos = -1  # will use original subtoken

        tasks.append({
            "input_ids": input_ids,
            "mask_pos": mask_pos,
            "eps_per_subtoken": eps_per_subtoken,
            "original_subtoken_id": sub_token_id,
        })

    return tasks, None


def _run_dpmlm_batch(dpmlm, all_tasks, filter_numeric=False, numeric_mask=None,
                     batch_size=64):
    """Run a batch of masked-subtoken predictions through the MLM.

    Args:
        dpmlm: DPMLM instance
        all_tasks: List of task dicts from _prepare_dpmlm_tasks
        filter_numeric: If True, mask non-numeric tokens before sampling
        numeric_mask: Boolean mask over vocabulary
        batch_size: Max batch size for GPU forward pass

    Returns:
        List of chosen token strings, one per task.
    """
    tokenizer = dpmlm.tokenizer
    results = [None] * len(all_tasks)

    # Handle tasks with no mask position (use original subtoken)
    pending_indices = []
    for i, task in enumerate(all_tasks):
        if task["mask_pos"] < 0:
            results[i] = tokenizer.decode([task["original_subtoken_id"]]).strip()
        else:
            pending_indices.append(i)

    if not pending_indices:
        return results

    # Process in batches
    for batch_start in range(0, len(pending_indices), batch_size):
        batch_idx = pending_indices[batch_start:batch_start + batch_size]
        batch_tasks = [all_tasks[i] for i in batch_idx]

        # Pad to same length
        max_len = max(len(t["input_ids"]) for t in batch_tasks)
        padded = []
        attention_masks = []
        for t in batch_tasks:
            ids = t["input_ids"]
            pad_len = max_len - len(ids)
            padded.append(ids + [tokenizer.pad_token_id] * pad_len)
            attention_masks.append([1] * len(ids) + [0] * pad_len)

        input_tensor = torch.tensor(padded, device=dpmlm.device)
        attn_tensor = torch.tensor(attention_masks, device=dpmlm.device)

        with torch.no_grad():
            output = dpmlm.lm_model(input_ids=input_tensor,
                                     attention_mask=attn_tensor)

        logits = output.logits.detach().cpu().float().numpy()

        for j, idx in enumerate(batch_idx):
            task = all_tasks[idx]
            mask_logits = logits[j, task["mask_pos"]]
            eps = task["eps_per_subtoken"]

            mask_logits = np.clip(mask_logits, dpmlm.clip_min, dpmlm.clip_max)
            mask_logits = mask_logits / (2 * dpmlm.sensitivity / eps)

            if filter_numeric and numeric_mask is not None:
                scaled_clip_min = dpmlm.clip_min / (2 * dpmlm.sensitivity / eps)
                mask_logits[~numeric_mask[:len(mask_logits)]] = scaled_clip_min

            scores = torch.softmax(torch.from_numpy(mask_logits), dim=0)
            scores = scores / scores.sum()
            chosen_idx = np.random.choice(len(scores), p=scores.numpy())
            results[idx] = tokenizer.decode([chosen_idx]).strip()

    return results


def _find_subsequence(full_seq, sub_seq):
    """Find the start index of sub_seq in full_seq, or None."""
    n = len(full_seq)
    m = len(sub_seq)
    for i in range(n - m + 1):
        if full_seq[i:i+m] == sub_seq:
            return i
    return None


def _parse_numeric(replacement_str):
    """Parse a string as a float.

    Returns:
        (value, is_numeric): tuple of parsed float and success flag.
    """
    if not replacement_str:
        return 0.0, False

    s = replacement_str.strip()

    # Direct parse
    try:
        return float(s), True
    except ValueError:
        pass

    # Regex fallback: extract first numeric substring
    match = re.search(r'-?\d+\.?\d*', s)
    if match:
        try:
            return float(match.group()), True
        except ValueError:
            pass

    return 0.0, False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def sanitize_sbyw_dpmlm(samples, nodes, edges, expressions, epsilon,
                         target_keys, template_name, filter_numeric=False,
                         use_discrete=False):
    """
    SBYW + DP-MLM baseline sanitization.

    Every non-classification node is sanitized independently using its ground
    truth value. Budget is distributed via SBYW importance scoring on the
    paragraph representation, then DP-MLM replaces numeric values at BPE level.

    Args:
        samples: List of sample dictionaries with conversation turns
        nodes: Dictionary mapping node names to their parent dependencies
        edges: List of (source, target) dependency edges
        expressions: Dictionary mapping node names to computation expressions
        epsilon: Base privacy budget per node
        target_keys: Dict mapping template name to target node name
        template_name: Name of the current template (e.g., "HOMA")
        filter_numeric: If True, use vocab-filtered DP-MLM (Version B).
                        If False, use raw DP-MLM with fallback (Version A).
        use_discrete: If True, use M_exponential_discrete for fallbacks
                      instead of M_exponential.

    Returns:
        (all_values, epsilon_blocks, fallback_log, prompt_log)
        - all_values: List of sanitized value dicts (same format as
          sanitize_inferdpt)
        - epsilon_blocks: List of per-node epsilon lists
        - fallback_log: List of dicts recording each fallback event
          (Version A only; empty for Version B)
        - prompt_log: List of dicts recording every input prompt sent
          to DP-MLM, keyed by sample index and node name
    """
    eps_dist = _get_eps_distributor()
    dpmlm = _get_dpmlm()
    san = _get_sanitizer()

    def _fallback_noise(value, eps, lo, hi):
        if use_discrete:
            return san.M_exponential_discrete(value, eps, lo, hi)
        return san.M_exponential_discrete(value, eps, lo, hi)

    numeric_mask = None
    if filter_numeric:
        numeric_mask = _get_numeric_vocab_mask(dpmlm.tokenizer)

    epsilon_blocks = []
    all_values = []
    fallback_log = []
    prompt_log = []

    for sample_idx, sample in enumerate(samples):
        topo_order, gt_values = get_topological_order(sample, edges)
        ground_truth = {k: v for k, v in gt_values.items()}
        sanitized_values = {}
        actual_budgets = {}
        sample_fallbacks = []

        # Count non-class nodes for total budget
        n_nodes = len([n for n in topo_order if n not in _SKIP_NODES])
        total_epsilon = n_nodes * epsilon

        # Step 1: Generate paragraph from ground truth
        paragraph = sample_to_paragraph(sample, template_name)

        # Step 2: SBYW importance scoring
        sbyw_result = eps_dist.get_distribution(paragraph, total_epsilon=total_epsilon)

        # Step 3: Extract numeric value weights
        value_node_pairs = _build_value_to_node_map(ground_truth, template_name)
        value_weights = _extract_value_weights(sbyw_result, value_node_pairs)

        # Step 4: Redistribute budget
        node_epsilons = _distribute_budget(value_weights, total_epsilon)

        # Step 5: Prepare all MLM tasks for this sample (batched)
        mlm_node_tasks = []  # list of (node, gt_val, node_eps, tasks, decimals)
        for node in topo_order:
            if node in _SKIP_NODES:
                sanitized_values[node] = ground_truth[node]
                actual_budgets[node] = 0
                continue

            if node not in nodes:
                sanitized_values[node] = ground_truth[node]
                actual_budgets[node] = 0
                continue

            node_eps = node_epsilons.get(node)
            if node_eps is None:
                node_eps = epsilon
                lo, hi = MEDICAL_DOMAINS.get(node, MEDICAL_DOMAINS["_default"])
                sanitized_values[node] = _fallback_noise(
                    float(ground_truth[node]), node_eps, lo, hi
                )
                actual_budgets[node] = node_eps
                continue

            order = _TEMPLATE_VALUE_ORDER.get(template_name, [])
            decimals = 2
            for n_name, n_dec in order:
                if n_name == node:
                    decimals = n_dec
                    break

            gt_val = ground_truth[node]
            if decimals == 0:
                value_str = str(int(gt_val))
            else:
                value_str = _fmt(gt_val, decimals)

            tasks, fallback_result = _prepare_dpmlm_tasks(
                dpmlm, paragraph, value_str, node_eps
            )

            if not tasks and fallback_result is not None:
                # Value string returned directly (empty BPE)
                parsed_val, is_numeric = _parse_numeric(fallback_result)
                lo, hi = MEDICAL_DOMAINS.get(node, MEDICAL_DOMAINS["_default"])
                if is_numeric:
                    sanitized_values[node] = float(np.clip(parsed_val, lo, hi))
                else:
                    sanitized_values[node] = _fallback_noise(
                        float(gt_val), node_eps, lo, hi
                    )
                actual_budgets[node] = node_eps
                continue

            if not tasks:
                # BPE subsequence not found — word-level fallback
                result = dpmlm.privatize(paragraph, value_str, 1, 0,
                                         CONCAT=True, epsilon=node_eps)
                key = f"{value_str}_1"
                replacement_str = result.get(key, value_str)
                parsed_val, is_numeric = _parse_numeric(replacement_str)
                lo, hi = MEDICAL_DOMAINS.get(node, MEDICAL_DOMAINS["_default"])
                if is_numeric:
                    sanitized_values[node] = float(np.clip(parsed_val, lo, hi))
                else:
                    sanitized_values[node] = _fallback_noise(
                        float(gt_val), node_eps, lo, hi
                    )
                    sample_fallbacks.append({
                        "sample_idx": sample_idx, "node": node,
                        "original_value": float(gt_val),
                        "dpmlm_output": replacement_str,
                        "fallback": True, "epsilon": node_eps,
                    })
                actual_budgets[node] = node_eps
                continue

            mlm_node_tasks.append((node, gt_val, node_eps, tasks))
            actual_budgets[node] = node_eps

        # Step 6: Run all MLM tasks for this sample in one batch
        if mlm_node_tasks:
            all_tasks = []
            task_node_map = []  # (node_idx, start, end) in all_tasks
            for ni, (node, gt_val, node_eps, tasks) in enumerate(mlm_node_tasks):
                start = len(all_tasks)
                all_tasks.extend(tasks)
                task_node_map.append((ni, start, start + len(tasks)))

            batch_results = _run_dpmlm_batch(
                dpmlm, all_tasks,
                filter_numeric=filter_numeric, numeric_mask=numeric_mask,
                batch_size=64
            )

            # Step 7: Assemble replacements per node
            for ni, start, end in task_node_map:
                node, gt_val, node_eps, tasks = mlm_node_tasks[ni]
                replacement_str = "".join(batch_results[start:end])
                parsed_val, is_numeric = _parse_numeric(replacement_str)
                lo, hi = MEDICAL_DOMAINS.get(node, MEDICAL_DOMAINS["_default"])

                if is_numeric:
                    sanitized_values[node] = float(np.clip(parsed_val, lo, hi))
                else:
                    sanitized_values[node] = _fallback_noise(
                        float(gt_val), node_eps, lo, hi
                    )
                    sample_fallbacks.append({
                        "sample_idx": sample_idx, "node": node,
                        "original_value": float(gt_val),
                        "dpmlm_output": replacement_str,
                        "fallback": True, "epsilon": node_eps,
                    })

        all_values.append(sanitized_values)
        epsilon_blocks.append(
            [actual_budgets.get(node, 0) for node in topo_order]
        )
        fallback_log.extend(sample_fallbacks)

    return all_values, epsilon_blocks, fallback_log, prompt_log
