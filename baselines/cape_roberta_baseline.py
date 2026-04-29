"""
CAPE Baseline — RoBERTa port (ICML 2025)

Reference: Wu et al., "CAPE: Context-Aware Prompt Perturbation Mechanism
with Differential Privacy", ICML 2025.

RoBERTa-base port of the CAPE mechanism with paper-correct hyperparameters:
  - λ_L = 0.5, λ_D = 1.0
  - Distance normalization: D_norm = (D - D_min)/(D_max - D_min),
    similarity = exp(-D_norm)  (Equations 4-6)
  - Logit clipping bound B calibrated per paper (Section 4, Appendix B.7):
    run masked predictions on calibration samples, set B = max |logit|.
  - N_b = 50 equal-width buckets

Differences from BERT version (cape_baseline.py):
  - Model: roberta-base (BPE tokenizer, no WordPiece ## artifacts)
  - Embeddings: .roberta.embeddings.word_embeddings
  - Special tokens: BOS/EOS instead of CLS/SEP
  - Logit clip: calibrated on data instead of hardcoded 10.0
"""

import re
import numpy as np

from preempt.sanitizer import MEDICAL_DOMAINS, Sanitizer
from utils.utils import get_topological_order
from baselines.dp_gtr_baseline import sample_to_paragraph

_SKIP_NODES = frozenset([
    "anemia_class", "fib4_risk", "aip_risk", "ci_risk",
    "ppi_risk", "tyg_class", "homa_class", "nlr_class",
    "const",  # mathematical constant (e.g., 405 in HOMA), not a private value
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

# Hyperparameters (from paper)
_NUM_BUCKETS = 50
_LAMBDA_L = 0.5       # paper: λ_L = 0.5
_LAMBDA_D = 1.0       # paper: λ_D = 1.0

# Lazy-loaded models
_mlm_model = None
_tokenizer = None
_static_embs = None
_vocab_size = None
_dist_cache = {}
_numeric_vocab_mask = None
_logit_clip = None    # calibrated at runtime


def _load_models():
    """Lazy-load RoBERTa MLM model, tokenizer, and static embeddings."""
    global _mlm_model, _tokenizer, _static_embs, _vocab_size

    if _mlm_model is not None:
        return

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model_name = "roberta-base"
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _mlm_model = AutoModelForMaskedLM.from_pretrained(model_name)
    _mlm_model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _mlm_model = _mlm_model.to(device)

    # Extract static word embeddings for distance computation
    _static_embs = (_mlm_model.roberta.embeddings.word_embeddings
                    .weight.detach().cpu().numpy())
    _vocab_size = _static_embs.shape[0]

    # Build numeric vocabulary mask (must contain at least one digit —
    # excludes pure dash/dot sequences like "---", "..." from RoBERTa BPE)
    global _numeric_vocab_mask
    numeric_re = re.compile(r'^[\d.\-]*\d[\d.\-]*$')
    _numeric_vocab_mask = np.zeros(_vocab_size, dtype=bool)
    for idx in range(_vocab_size):
        decoded = _tokenizer.decode([idx]).strip()
        if decoded and numeric_re.match(decoded):
            _numeric_vocab_mask[idx] = True
    n_numeric_tokens = int(_numeric_vocab_mask.sum())

    print(f"    [CAPE-RoBERTa] Loaded on {device}, vocab_size={_vocab_size}, "
          f"numeric_tokens={n_numeric_tokens}", flush=True)


def _calibrate_logit_clip(n_values_per_node=20):
    """Calibrate the logit clipping bound B per the CAPE paper.

    Per Section 4 and Appendix B.7: analyze the logit distribution by
    running masked predictions on calibration data and setting B to the
    maximum observed absolute logit value.

    Uses public domain bounds (MEDICAL_DOMAINS) to generate synthetic
    numeric strings — no private patient data used. For each measurement
    node, samples values uniformly from [lo, hi], formats them as strings,
    tokenizes, and runs MLM at each token position.

    Args:
        n_values_per_node: Number of synthetic values per domain
    """
    global _logit_clip

    if _logit_clip is not None:
        return

    import torch

    _load_models()
    device = next(_mlm_model.parameters()).device

    bos_id = _tokenizer.bos_token_id
    eos_id = _tokenizer.eos_token_id
    mask_id = _tokenizer.mask_token_id

    # Generate synthetic numeric strings from public domain bounds
    calibration_strings = []
    for node, (lo, hi) in MEDICAL_DOMAINS.items():
        if node == "_default":
            continue
        values = np.linspace(lo, hi, n_values_per_node)
        for v in values:
            calibration_strings.append(f"{v:.2f}")

    print(f"    [CAPE-RoBERTa] Calibrating logit clip B on "
          f"{len(calibration_strings)} synthetic values from domain bounds...",
          flush=True)

    max_abs_logit = 0.0
    for val_str in calibration_strings:
        token_ids = _tokenizer.encode(val_str, add_special_tokens=False)
        if not token_ids:
            continue

        # Run MLM at each token position
        batch = []
        positions = list(range(len(token_ids)))
        for pos in positions:
            ids = list(token_ids)
            ids[pos] = mask_id
            batch.append([bos_id] + ids + [eos_id])

        input_tensor = torch.tensor(batch, dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = _mlm_model(input_tensor)
            for i, pos in enumerate(positions):
                logits = outputs.logits[i, pos + 1, :].cpu().numpy()
                sample_max = float(np.max(np.abs(logits)))
                if sample_max > max_abs_logit:
                    max_abs_logit = sample_max

    _logit_clip = max_abs_logit
    print(f"    [CAPE-RoBERTa] Calibrated B = {_logit_clip:.4f}", flush=True)


def _get_mlm_logits_batched(token_ids, mask_positions):
    """Get MLM logits at multiple masked positions in a single batched pass.

    Creates one copy of the sequence per mask position, each with a different
    token masked, and runs them all through the model at once.

    Args:
        token_ids: List of token IDs (without BOS/EOS)
        mask_positions: List of int positions to mask

    Returns:
        numpy array of shape (len(mask_positions), vocab_size)
    """
    import torch

    _load_models()
    device = next(_mlm_model.parameters()).device

    bos_id = _tokenizer.bos_token_id
    eos_id = _tokenizer.eos_token_id
    mask_id = _tokenizer.mask_token_id

    batch = []
    for pos in mask_positions:
        ids = list(token_ids)
        ids[pos] = mask_id
        batch.append([bos_id] + ids + [eos_id])

    input_tensor = torch.tensor(batch, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = _mlm_model(input_tensor)
        all_logits = []
        for i, pos in enumerate(mask_positions):
            all_logits.append(outputs.logits[i, pos + 1, :].cpu().numpy())

    return np.array(all_logits)


def _cape_select(logits, target_id, epsilon):
    """CAPE bucketized selection for one token (paper-correct).

    Hybrid utility: u(r) = clip(L_r)^λ_L · sim(t_i, r)^λ_D
    Distance: D_norm = (D - D_min)/(D_max - D_min), sim = exp(-D_norm)

    Returns:
        Replacement token ID
    """
    clipped = np.clip(logits, -_logit_clip, _logit_clip)
    shifted = clipped + _logit_clip  # ∈ [0, 2B]

    if target_id in _dist_cache:
        similarity = _dist_cache[target_id]
    else:
        static_emb = _static_embs[target_id]
        dists = np.linalg.norm(_static_embs - static_emb[np.newaxis, :], axis=1)
        d_min = dists.min()
        d_max = dists.max()
        d_norm = (dists - d_min) / (d_max - d_min + 1e-10)
        similarity = np.exp(-d_norm)
        _dist_cache[target_id] = similarity

    utility = (shifted ** _LAMBDA_L) * (similarity ** _LAMBDA_D)

    u_min, u_max = utility.min(), utility.max()
    bucket_width = (u_max - u_min + 1e-10) / _NUM_BUCKETS
    bucket_ids = np.clip(
        ((utility - u_min) / bucket_width).astype(int),
        0, _NUM_BUCKETS - 1
    )

    bucket_sums = np.bincount(bucket_ids, weights=utility,
                              minlength=_NUM_BUCKETS)
    bucket_sizes = np.bincount(bucket_ids, minlength=_NUM_BUCKETS)
    bucket_means = np.where(bucket_sizes > 0, bucket_sums / bucket_sizes, 0.0)

    # Paper Theorem 4.1: (ε + ε')-DP; Algorithm 2 uses ε directly
    nonempty = bucket_sizes > 0

    delta = (2.0 * _logit_clip) ** _LAMBDA_L
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
    members = np.where(bucket_ids == selected_bucket)[0]
    return int(np.random.choice(members))


def _cape_select_filtered(logits, target_id, epsilon):
    """CAPE bucketized selection restricted to numeric-only vocabulary.

    Same mechanism as _cape_select but zeroes out utility for non-numeric
    tokens before bucketizing, guaranteeing numeric output.
    """
    clipped = np.clip(logits, -_logit_clip, _logit_clip)
    shifted = clipped + _logit_clip

    if target_id in _dist_cache:
        similarity = _dist_cache[target_id]
    else:
        static_emb = _static_embs[target_id]
        dists = np.linalg.norm(_static_embs - static_emb[np.newaxis, :], axis=1)
        d_min = dists.min()
        d_max = dists.max()
        d_norm = (dists - d_min) / (d_max - d_min + 1e-10)
        similarity = np.exp(-d_norm)
        _dist_cache[target_id] = similarity

    utility = (shifted ** _LAMBDA_L) * (similarity ** _LAMBDA_D)
    utility[~_numeric_vocab_mask] = 0.0

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

    delta = (2.0 * _logit_clip) ** _LAMBDA_L
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
    """Map each node's formatted value to its BPE token positions.

    Decodes the full token sequence into a character string, keeping track
    of which characters map to which token indices. For each node value,
    finds its formatted string and maps back to token indices.

    Returns:
        List of (node_name, start_token, end_token). (-1, -1) if not found.
    """
    char_to_token = []
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

        char_pos = full_text.find(val_str, search_from_char)
        if char_pos >= 0:
            char_end = char_pos + len(val_str)
            start_token = char_to_token[char_pos]
            end_token = char_to_token[min(char_end - 1, len(char_to_token) - 1)] + 1
            spans.append((node_name, start_token, end_token))
            search_from_char = char_end
        else:
            spans.append((node_name, -1, -1))

    return spans


def sanitize_cape_roberta(samples, nodes, edges, epsilon, template_target_keys,
                          template_name=None):
    """Sanitize medical values via CAPE with RoBERTa (vanilla, all vocab)."""
    _load_models()
    _calibrate_logit_clip()

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
                    new_id = _cape_select(all_logits[midx], orig_tid, eps_per_tok)
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
            print(f"      [CAPE-RoBERTa] {si+1}/{len(samples)} samples done",
                  flush=True)

    if total_values > 0:
        rate = fallback_count / total_values * 100
        print(f"    [CAPE-RoBERTa] Fallback rate: {fallback_count}/{total_values} "
              f"({rate:.1f}%)", flush=True)

    return all_values, epsilon_blocks


def sanitize_cape_roberta_filtered(samples, nodes, edges, epsilon,
                                   template_target_keys, template_name=None):
    """CAPE-F with RoBERTa: numeric-filtered vocabulary.

    Tokenizes each value directly (no paragraph search), runs MLM at each
    token position for context-aware logits, and applies the filtered CAPE
    mechanism. Every node is processed — no fallback needed.

    Budget: each node gets uniform epsilon, split across BPE tokens.
    """
    _load_models()
    _calibrate_logit_clip()

    epsilon_blocks = []
    all_values = []
    total_values = 0

    value_order = _TEMPLATE_VALUE_ORDER.get(template_name, [])
    node_decimals = {name: dec for name, dec in value_order}

    for si, sample in enumerate(samples):
        topo_order, gt_values = get_topological_order(sample, edges)
        nodes_to_release = [n for n in topo_order if n not in _SKIP_NODES]

        sanitized = {}
        budgets = []
        for node in nodes_to_release:
            total_values += 1
            domain_lower, domain_upper = MEDICAL_DOMAINS.get(
                node, MEDICAL_DOMAINS["_default"])
            true_val = float(gt_values[node])

            # Format and tokenize the value directly
            decimals = node_decimals.get(node, 2)
            if decimals == 0:
                val_str = str(int(true_val))
            else:
                val_str = f"{true_val:.{decimals}f}"

            token_ids = _tokenizer.encode(val_str, add_special_tokens=False)

            # Structural tokens ("." decimal point, "-" sign) are preserved
            # as-is. Only digit tokens consume privacy budget.
            sanitizable = [(i, tid) for i, tid in enumerate(token_ids)
                           if _tokenizer.decode([tid]).strip() not in ('.', '-')]
            n_sanitizable = len(sanitizable)
            eps_per_tok = epsilon / max(n_sanitizable, 1)

            # Get MLM logits only at sanitizable positions
            if sanitizable:
                mask_positions = [i for i, _ in sanitizable]
                logits = _get_mlm_logits_batched(token_ids, mask_positions)
            else:
                logits = np.empty((0, _vocab_size))

            # Build replacement: keep dots, replace digits via CAPE
            replacement_ids = list(token_ids)  # start with original
            for li, (pos, orig_tid) in enumerate(sanitizable):
                new_id = _cape_select_filtered(logits[li], orig_tid, eps_per_tok)
                replacement_ids[pos] = new_id

            # Decode and parse
            replacement_str = _tokenizer.decode(replacement_ids,
                                                skip_special_tokens=True).strip()
            match = _NUMBER_PATTERN.search(replacement_str)
            if match:
                san_val = float(match.group())
            else:
                san_val = true_val  # should not happen with dots preserved

            san_val = np.clip(san_val, domain_lower, domain_upper)
            sanitized[node] = round(float(san_val), ndigits=4)
            budgets.append(epsilon)

        all_values.append(sanitized)
        epsilon_blocks.append(budgets)

        if (si + 1) % 100 == 0:
            print(f"      [CAPE-RoBERTa-F] {si+1}/{len(samples)} samples done",
                  flush=True)

    print(f"    [CAPE-RoBERTa-F] Processed {total_values} values, 0 fallbacks",
          flush=True)

    return all_values, epsilon_blocks


# ---------------------------------------------------------------------------
# Generic CAPE perturbation for text classification (SST-2 evaluation)
# ---------------------------------------------------------------------------

def cape_perturb_text(text, epsilon, non_sensitive_words=None):
    """Apply CAPE perturbation to arbitrary text for classification tasks.

    Perturbs all tokens except those in non_sensitive_words (stopwords,
    punctuation). This is the general-purpose CAPE mechanism as described
    in the paper, not specific to medical numeric sanitization.

    Args:
        text: Input text string
        epsilon: Privacy budget (total, split across sensitive tokens)
        non_sensitive_words: Set of words to skip (default: NLTK stopwords
            + punctuation, as per paper)

    Returns:
        Perturbed text string
    """
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    _load_models()

    if non_sensitive_words is None:
        from nltk.corpus import stopwords
        non_sensitive_words = set(stopwords.words('english'))
        non_sensitive_words.update(set('.,?!%()/-:;"\' '))

    # Calibrate logit clip if not done
    _calibrate_logit_clip()

    token_ids = _tokenizer.encode(text, add_special_tokens=False)

    # Identify sensitive token positions
    sensitive_positions = []
    for pos, tid in enumerate(token_ids):
        decoded = _tokenizer.decode([tid]).strip().lower()
        if decoded and decoded not in non_sensitive_words:
            sensitive_positions.append(pos)

    if not sensitive_positions:
        return text

    # Paper: ε-LDP per token (each token gets full budget independently)
    eps_per_tok = epsilon

    # Batch MLM logits for all sensitive positions
    all_logits = _get_mlm_logits_batched(token_ids, sensitive_positions)

    # Replace each sensitive token via CAPE mechanism
    new_token_ids = list(token_ids)
    for i, pos in enumerate(sensitive_positions):
        orig_tid = token_ids[pos]
        new_id = _cape_select(all_logits[i], orig_tid, eps_per_tok)
        new_token_ids[pos] = new_id

    return _tokenizer.decode(new_token_ids, skip_special_tokens=True)


