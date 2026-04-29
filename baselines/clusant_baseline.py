"""
CluSanT Baselines (NAACL 2025)

Reference: Awon et al., "CluSanT: Differentially Private and Semantically
Coherent Text Sanitization", NAACL 2025.
Code: https://github.com/AwonSomeSauce/CluSanT

Two variants:

  CluSanT-vanilla: Faithful text-level adaptation using the original
    CluSanT mechanism with all-mpnet-base-v2 embeddings. Builds a
    vocabulary of all words/numbers in medical paragraphs, clusters them,
    and applies the two-level exponential mechanism. Demonstrates that
    text-level DP methods fail for numeric data.

  CluSanT-numeric: Applies CluSanT's two-level exponential mechanism
    directly to the numeric domain. For each measurement, discretizes
    the medical domain into candidates, clusters them by numeric
    proximity, and applies the same two-level selection (cluster → value).
    Tests whether CluSanT's hierarchical structure helps for numeric data.

Privacy guarantee: ε-metric-LDP per token/value (basic composition).
"""

import re
import numpy as np
from scipy.spatial.distance import cdist, pdist

from preempt.sanitizer import MEDICAL_DOMAINS, Sanitizer
from utils.utils import get_topological_order
from baselines.dp_gtr_baseline import sample_to_paragraph, _fmt


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


# ============================================================================
# CluSanT-vanilla: faithful text-level adaptation
# ============================================================================

_vanilla_clusant = None      # CluSanT instance (lazy)
_vanilla_embeddings = None   # word → embedding dict
_vanilla_word_to_cluster = None  # word → cluster_label lookup


def _build_public_vocabulary(num_candidates=1000):
    """Build a PUBLIC, data-independent vocabulary from medical domain grids.

    For each node in MEDICAL_DOMAINS, generates num_candidates evenly spaced
    values across [domain_lower, domain_upper] and formats them as strings.
    This is the same grid used by M_exponential_discrete.

    Also includes non-numeric text labels from paragraph templates (public
    knowledge: metric names, units, question text).
    """
    words = set()

    # Numeric candidates: same grid as the discrete exponential mechanism
    for node, (lo, hi) in MEDICAL_DOMAINS.items():
        if node == "_default":
            continue
        candidates = np.linspace(lo, hi, num_candidates)
        # Format to reasonable precision (4 decimal places, strip trailing zeros)
        for v in candidates:
            # Use multiple format precisions to cover how values appear in paragraphs
            words.add(f"{v:.0f}")
            words.add(f"{v:.1f}")
            words.add(f"{v:.2f}")
            words.add(f"{v:.4f}")

    # Non-numeric text labels from paragraph templates (public, not data-dependent)
    _PUBLIC_TEXT = (
        "my hemoglobin level is g dl and hematocrit red blood cell count million "
        "mean corpuscular volume mcv mch concentration mchc what anemia classification "
        "total cholesterol mg hdl non triglycerides ldl atherogenic index plasma aip "
        "cardiovascular risk weight kg height m bmi waist circumference ratio body shape "
        "systolic pressure mmhg diastolic pulse arterial map mid mbp ppi vascular status "
        "age year old ast alt plt fib4 liver fibrosis score product denominator "
        "fasting glucose insulin homa resistance tyg triglyceride "
        "neutrophil lymphocyte nlr sum diff immune ratio"
    )
    for w in _PUBLIC_TEXT.lower().split():
        words.add(w)
    # Punctuation
    for p in ".,?!%()/-":
        words.add(p)

    return words


def _init_vanilla_clusant(all_samples_unused, epsilon, num_clusters=40, K=8):
    """Initialize the CluSanT mechanism for text-level sanitization.

    Vocabulary is PUBLIC: numeric candidates from MEDICAL_DOMAINS grids
    (same as the discrete exponential mechanism) plus non-numeric text labels.
    No private data is used in vocabulary construction.

    Uses all-mpnet-base-v2 embeddings (same as original paper) and
    the CluSanT greedy clustering algorithm.
    """
    global _vanilla_clusant, _vanilla_embeddings, _vanilla_word_to_cluster

    if _vanilla_clusant is not None:
        # Update epsilon if changed
        _vanilla_clusant.epsilon = epsilon
        return

    from sentence_transformers import SentenceTransformer

    print("    [CluSanT-vanilla] Building PUBLIC vocabulary from domain grids...",
          flush=True)

    words = _build_public_vocabulary(num_candidates=1000)
    print(f"    [CluSanT-vanilla] Vocabulary size: {len(words)}", flush=True)

    # Generate embeddings using all-mpnet-base-v2
    print("    [CluSanT-vanilla] Encoding embeddings...", flush=True)
    model = SentenceTransformer("all-mpnet-base-v2")
    word_list = sorted(words)
    # Batch encode in chunks to avoid OOM on large vocabs
    chunk_size = 1024
    emb_chunks = []
    for i in range(0, len(word_list), chunk_size):
        chunk = word_list[i:i + chunk_size]
        emb_chunks.append(model.encode(chunk, show_progress_bar=False,
                                       batch_size=256))
        if (i // chunk_size) % 50 == 0:
            print(f"      encoded {i + len(chunk)}/{len(word_list)}", flush=True)
    emb_matrix = np.vstack(emb_chunks)
    _vanilla_embeddings = {w: emb_matrix[i].tolist()
                           for i, w in enumerate(word_list)}
    del model

    # Use CluSanT's own clustering (imported from the cloned repo)
    import sys, os
    sys.path.insert(0, "CluSanT/src")
    from clusant import CluSanT as CluSanTOriginal

    # Ensure cache directories exist for CluSanT's file-based caching
    for d in ["clusters", "centroids", "intra", "inter"]:
        os.makedirs(d, exist_ok=True)

    # Clear stale cache from previous vocabulary
    import glob
    for pattern in ["clusters/medical_*", "centroids/medical_*",
                    "intra/medical_*", "inter/medical_*"]:
        for f in glob.glob(pattern):
            os.remove(f)

    # Adjust num_clusters: ensure at least ~3 words per cluster
    actual_clusters = min(num_clusters, max(1, len(word_list) // 3))

    print(f"    [CluSanT-vanilla] Clustering into {actual_clusters} clusters...",
          flush=True)
    _vanilla_clusant = CluSanTOriginal(
        embedding_file="medical_vocab_mpnet",
        embeddings=_vanilla_embeddings,
        epsilon=epsilon,
        num_clusters=actual_clusters,
        mechanism="clusant",
        dp_type="metric",
        K=K,
    )

    # Build fast word → cluster lookup
    _vanilla_word_to_cluster = {}
    for label, members in _vanilla_clusant.clusters.items():
        for w in members:
            _vanilla_word_to_cluster[w] = label

    # Pre-compute numpy arrays for fast replacement (bypass cdist per call)
    global _cluster_emb_arrays, _cluster_word_arrays, _centroid_array
    _cluster_emb_arrays = {}  # cluster_id → (n_members, embed_dim) numpy array
    _cluster_word_arrays = {}  # cluster_id → list of word strings
    for label, members in _vanilla_clusant.clusters.items():
        _cluster_word_arrays[label] = members
        _cluster_emb_arrays[label] = np.array(
            [_vanilla_embeddings[w] for w in members], dtype=np.float32)
    _centroid_array = np.array(
        [_vanilla_clusant.inter_distances[0] for _ in range(actual_clusters)],
        dtype=np.float32)  # placeholder, we use the actual inter_distances matrix

    print(f"    [CluSanT-vanilla] Created {actual_clusters} clusters, "
          f"pre-computed arrays, ready to sanitize.", flush=True)


def _fast_replace_word(word_str, epsilon):
    """Fast CluSanT two-level replacement using pre-computed numpy arrays.

    Bypasses the original CluSanT.replace_word() which does linear scan +
    per-call cdist. Uses cached cluster embedding arrays instead.
    """
    word_str = word_str.lower()
    cluster_label = _vanilla_word_to_cluster.get(word_str)
    if cluster_label is None:
        return None

    n_clusters = len(_cluster_emb_arrays)

    # Level 1: Select cluster via exponential mechanism over inter-cluster distances
    if n_clusters > 1:
        inter_dists = _vanilla_clusant.inter_distances[cluster_label]
        utilities = -inter_dists
        log_probs = epsilon * utilities / (2.0 * _vanilla_clusant.inter_cluster_sensitivity)
        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        probs /= probs.sum()
        selected = np.random.choice(n_clusters, p=probs)
    else:
        selected = cluster_label

    # Level 2: Select word within cluster via exponential mechanism
    cluster_embs = _cluster_emb_arrays[selected]  # (n_members, dim)
    word_emb = np.array(_vanilla_embeddings[word_str], dtype=np.float32)
    dists = np.linalg.norm(cluster_embs - word_emb[np.newaxis, :], axis=1)

    intra_sens = _vanilla_clusant.intra_cluster_sensitivity[selected]
    if intra_sens <= 0:
        intra_sens = 1e-10

    utilities2 = -dists
    log_probs2 = epsilon * utilities2 / (2.0 * intra_sens)
    log_probs2 -= np.max(log_probs2)
    probs2 = np.exp(log_probs2)
    probs2 /= probs2.sum()

    idx = np.random.choice(len(probs2), p=probs2)
    return _cluster_word_arrays[selected][idx]


def _snap_to_vocab(tok):
    """Map a numeric token to the nearest string in the public vocabulary.

    Tries multiple format precisions (0-4 decimal places) and returns the
    first match found in the vocabulary. Falls back to the original token.
    """
    try:
        val = float(tok)
    except ValueError:
        return tok

    for fmt in [f"{val:.0f}", f"{val:.1f}", f"{val:.2f}", f"{val:.4f}"]:
        if _vanilla_embeddings is not None and fmt in _vanilla_embeddings:
            return fmt
    # If no exact format match, return 2-decimal (most common in grid)
    return f"{val:.2f}"


def sanitize_clusant_vanilla(samples, nodes, edges, epsilon,
                             template_target_keys, template_name=None,
                             all_samples=None):
    """
    CluSanT-vanilla: text-level DP using CluSanT's two-level exponential
    mechanism over a public vocabulary of numeric strings.

    For each non-skip node: format value as string → snap to nearest
    vocabulary entry → CluSanT replace_word() → parse replacement as float.

    Budget: each node gets uniform epsilon (same total as vanilla).
    No DAG awareness, no post-processing.
    """
    _init_vanilla_clusant(None, epsilon)
    _vanilla_clusant.epsilon = epsilon

    san = Sanitizer()
    epsilon_blocks = []
    all_values = []
    fallback_count = 0
    total_values = 0

    for sample in samples:
        topo_order, values = get_topological_order(sample, edges)
        nodes_to_release = [n for n in topo_order if n not in _SKIP_NODES]
        sanitized = {}
        budgets = []

        for node in nodes_to_release:
            total_values += 1
            true_val = float(values[node])
            domain_lower, domain_upper = MEDICAL_DOMAINS.get(
                node, MEDICAL_DOMAINS["_default"])

            # Format value as string and snap to nearest vocabulary entry
            lookup_str = _snap_to_vocab(f"{true_val:.4f}")

            replacement = _fast_replace_word(lookup_str, epsilon)

            # Parse replacement back to float
            if replacement is not None:
                try:
                    san_val = float(replacement)
                    san_val = np.clip(san_val, domain_lower, domain_upper)
                    sanitized[node] = round(float(san_val), ndigits=4)
                except ValueError:
                    # Replacement is non-numeric text → fallback
                    fallback_count += 1
                    noised = san.M_exponential_discrete(
                        true_val, epsilon, domain_lower, domain_upper)
                    sanitized[node] = round(float(noised), ndigits=4)
            else:
                # Word not in vocabulary → fallback
                fallback_count += 1
                noised = san.M_exponential_discrete(
                    true_val, epsilon, domain_lower, domain_upper)
                sanitized[node] = round(float(noised), ndigits=4)

            budgets.append(epsilon)

        all_values.append(sanitized)
        epsilon_blocks.append(budgets)

    if total_values > 0:
        rate = fallback_count / total_values * 100
        print(f"    [CluSanT-vanilla] Fallback rate: "
              f"{fallback_count}/{total_values} ({rate:.1f}%)", flush=True)

    return all_values, epsilon_blocks


# ============================================================================
# CluSanT-numeric: two-level exponential mechanism on numeric domains
# ============================================================================

# Per-node cached clustering structure
_numeric_clusters = {}  # node → {clusters, centroids, inter_dists, intra_sens, candidates}


def _build_numeric_clusters(node, domain_lower, domain_upper,
                            num_candidates=1000, num_clusters=20):
    """Build numeric clusters for a measurement node.

    Discretizes [domain_lower, domain_upper] into num_candidates values,
    then partitions into num_clusters equal-width contiguous bins.
    """
    if node in _numeric_clusters:
        return _numeric_clusters[node]

    candidates = np.linspace(domain_lower, domain_upper, num_candidates)

    # Equal-width contiguous clustering (most natural for 1D numeric data)
    cluster_size = max(1, num_candidates // num_clusters)
    clusters = {}       # cluster_id → array of candidate values
    centroids = []      # cluster centroids (mean value)

    for c in range(num_clusters):
        start = c * cluster_size
        end = min((c + 1) * cluster_size, num_candidates)
        if start >= num_candidates:
            break
        members = candidates[start:end]
        clusters[c] = members
        centroids.append(np.mean(members))

    centroids = np.array(centroids)
    actual_clusters = len(clusters)

    # Inter-cluster distances: |centroid_i - centroid_j|
    inter_dists = np.abs(centroids[:, None] - centroids[None, :])
    # Metric DP: sensitivity = 1
    inter_sensitivity = 1.0

    # Intra-cluster sensitivities: max pairwise distance within each cluster
    intra_sens = {}
    for c, members in clusters.items():
        if len(members) > 1:
            intra_sens[c] = float(members[-1] - members[0])  # contiguous, so max - min
        else:
            intra_sens[c] = 1e-10  # avoid division by zero

    result = {
        "clusters": clusters,
        "centroids": centroids,
        "inter_dists": inter_dists,
        "inter_sensitivity": inter_sensitivity,
        "intra_sens": intra_sens,
        "candidates": candidates,
        "n_clusters": actual_clusters,
    }
    _numeric_clusters[node] = result
    return result


def _clusant_numeric_mechanism(true_val, epsilon, node, domain_lower, domain_upper,
                               num_candidates=1000, num_clusters=20):
    """
    CluSanT two-level exponential mechanism on numeric domain.

    Level 1: Select cluster via exponential mechanism over cluster centroids.
      prob(cluster c) ∝ exp(ε/2 · (-|true_val - centroid_c|) / Δ_inter)

    Level 2: Select candidate within cluster via exponential mechanism.
      prob(candidate v) ∝ exp(ε/2 · (-|true_val - v|) / Δ_intra(c))

    Args:
        true_val: True measurement value
        epsilon: Privacy budget for this value
        node: Node name (for caching cluster structure)
        domain_lower, domain_upper: Medical domain bounds
        num_candidates: Discretization resolution
        num_clusters: Number of numeric clusters

    Returns:
        Sanitized numeric value
    """
    info = _build_numeric_clusters(node, domain_lower, domain_upper,
                                   num_candidates, num_clusters)

    centroids = info["centroids"]
    n_clusters = info["n_clusters"]

    # Level 1: Select cluster
    centroid_dists = np.abs(true_val - centroids)
    utilities = -centroid_dists
    log_probs = epsilon * utilities / (2.0 * info["inter_sensitivity"])
    log_probs -= np.max(log_probs)
    probs = np.exp(log_probs)
    probs /= probs.sum()
    selected_cluster = np.random.choice(n_clusters, p=probs)

    # Level 2: Select value within cluster
    members = info["clusters"][selected_cluster]
    member_dists = np.abs(true_val - members)
    utilities2 = -member_dists
    sens = info["intra_sens"][selected_cluster]
    log_probs2 = epsilon * utilities2 / (2.0 * sens)
    log_probs2 -= np.max(log_probs2)
    probs2 = np.exp(log_probs2)
    probs2 /= probs2.sum()
    selected_idx = np.random.choice(len(members), p=probs2)

    return float(members[selected_idx])


def sanitize_clusant_numeric(samples, nodes, edges, bi_lipschitz_constants,
                             epsilon, num_candidates=1000, num_clusters=20):
    """
    CluSanT-numeric: two-level exponential mechanism on numeric domains.

    Each non-classification node gets uniform budget ε. Values are
    discretized, clustered by numeric proximity, and sanitized via
    CluSanT's two-level mechanism. No DAG awareness.

    Args:
        samples: List of sample dicts
        nodes: Template node definitions
        edges: Template edge definitions
        bi_lipschitz_constants: Not used (API compatibility)
        epsilon: Per-node privacy budget
        num_candidates: Discretization resolution (default 1000)
        num_clusters: Number of numeric clusters per node (default 20)

    Returns:
        (all_values, epsilon_blocks) matching vanilla format
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
                node, MEDICAL_DOMAINS["_default"])
            true_val = float(values[node])

            sanitized_val = _clusant_numeric_mechanism(
                true_val, epsilon, node, domain_lower, domain_upper,
                num_candidates, num_clusters)

            sanitized[node] = round(sanitized_val, ndigits=4)
            budgets.append(epsilon)

        all_values.append(sanitized)
        epsilon_blocks.append(budgets)

    return all_values, epsilon_blocks


# ============================================================================
# CluSanT-F: Filtered, embedding-based variant using RoBERTa tokens + MPNet
# ============================================================================

# Lazy-loaded state for CluSanT-F
_filtered_roberta_tok = None
_filtered_mpnet_embeddings = None    # dict: decoded_str → embedding list
_filtered_numeric_ids = None         # np.ndarray of RoBERTa numeric token IDs
_filtered_str_to_tid = None          # dict: decoded_str → token_id
_filtered_tid_to_str = None          # dict: token_id → decoded_str
_filtered_clusant = None             # CluSanT instance
_filtered_word_to_cluster = None     # str → cluster_label lookup
_filtered_cluster_emb_arrays = None  # cluster_id → (n, embed_dim) np array
_filtered_cluster_word_arrays = None # cluster_id → list of decoded strings


def _init_filtered_clusant(epsilon, num_clusters=40, K=8):
    """Initialize CluSanT-F: cluster numeric RoBERTa tokens by MPNet embedding.

    Uses RoBERTa tokenizer to identify numeric tokens, MPNet to embed them,
    and CluSanT's greedy clustering algorithm to group them.
    """
    global _filtered_roberta_tok, _filtered_mpnet_embeddings
    global _filtered_numeric_ids, _filtered_str_to_tid, _filtered_tid_to_str
    global _filtered_clusant, _filtered_word_to_cluster
    global _filtered_cluster_emb_arrays, _filtered_cluster_word_arrays

    if _filtered_clusant is not None:
        _filtered_clusant.epsilon = epsilon
        return

    import os as _os
    import glob as _glob
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer

    # Load RoBERTa tokenizer
    _filtered_roberta_tok = AutoTokenizer.from_pretrained("roberta-base")
    print("    [CluSanT-F] Loaded RoBERTa tokenizer", flush=True)

    # Get numeric token IDs
    from baselines.text_dp_common import build_roberta_numeric_mask
    mask = build_roberta_numeric_mask()
    _filtered_numeric_ids = np.where(mask)[0]

    # Decode each token and build mappings
    _filtered_str_to_tid = {}
    _filtered_tid_to_str = {}
    decoded_list = []
    for tid in _filtered_numeric_ids:
        decoded = _filtered_roberta_tok.decode([tid]).strip()
        _filtered_str_to_tid[decoded] = int(tid)
        _filtered_tid_to_str[int(tid)] = decoded
        decoded_list.append(decoded)

    print(f"    [CluSanT-F] {len(decoded_list)} numeric tokens identified",
          flush=True)

    # Encode with MPNet (CluSanT's embedding model)
    print("    [CluSanT-F] Encoding with all-mpnet-base-v2...", flush=True)
    mpnet = SentenceTransformer("all-mpnet-base-v2")
    emb_matrix = mpnet.encode(decoded_list, show_progress_bar=False,
                               batch_size=256)
    _filtered_mpnet_embeddings = {
        decoded_list[i]: emb_matrix[i].tolist()
        for i in range(len(decoded_list))
    }
    del mpnet

    # Cluster using CluSanT's algorithm
    import sys
    sys.path.insert(0, "CluSanT/src")
    from clusant import CluSanT as CluSanTOriginal

    # Ensure cache directories exist
    for d in ["clusters", "centroids", "intra", "inter"]:
        _os.makedirs(d, exist_ok=True)

    # Clear stale cache for this specific embedding file
    for pattern in ["clusters/roberta_numeric_mpnet_filtered_*",
                    "centroids/roberta_numeric_mpnet_filtered_*",
                    "intra/roberta_numeric_mpnet_filtered_*",
                    "inter/roberta_numeric_mpnet_filtered_*"]:
        for f in _glob.glob(pattern):
            _os.remove(f)

    actual_clusters = min(num_clusters, max(1, len(decoded_list) // 3))

    print(f"    [CluSanT-F] Clustering into {actual_clusters} clusters...",
          flush=True)
    _filtered_clusant = CluSanTOriginal(
        embedding_file="roberta_numeric_mpnet_filtered",
        embeddings=_filtered_mpnet_embeddings,
        epsilon=epsilon,
        num_clusters=actual_clusters,
        mechanism="clusant",
        dp_type="metric",
        K=K,
    )

    # Build fast lookups
    _filtered_word_to_cluster = {}
    for label, members in _filtered_clusant.clusters.items():
        for w in members:
            _filtered_word_to_cluster[w] = label

    # Pre-compute numpy arrays per cluster
    _filtered_cluster_emb_arrays = {}
    _filtered_cluster_word_arrays = {}
    for label, members in _filtered_clusant.clusters.items():
        _filtered_cluster_word_arrays[label] = members
        _filtered_cluster_emb_arrays[label] = np.array(
            [_filtered_mpnet_embeddings[w] for w in members], dtype=np.float32)

    print(f"    [CluSanT-F] Created {actual_clusters} clusters, ready.",
          flush=True)


def _filtered_replace_token(token_id, epsilon):
    """CluSanT two-level exponential mechanism for a single token.

    Level 1: Select cluster via inter-cluster distances.
    Level 2: Select token within cluster via intra-cluster distances.

    Returns:
        Replacement token ID, or None if token not in vocabulary.
    """
    decoded = _filtered_tid_to_str.get(int(token_id))
    if decoded is None or decoded not in _filtered_word_to_cluster:
        return None

    cluster_label = _filtered_word_to_cluster[decoded]
    n_clusters = len(_filtered_cluster_emb_arrays)

    # Level 1: Select cluster
    if n_clusters > 1:
        inter_dists = _filtered_clusant.inter_distances[cluster_label]
        utilities = -inter_dists
        log_probs = (epsilon * utilities /
                     (2.0 * _filtered_clusant.inter_cluster_sensitivity))
        log_probs -= np.max(log_probs)
        probs = np.exp(log_probs)
        probs /= probs.sum()
        selected = np.random.choice(n_clusters, p=probs)
    else:
        selected = cluster_label

    # Level 2: Select token within cluster
    cluster_embs = _filtered_cluster_emb_arrays[selected]
    word_emb = np.array(_filtered_mpnet_embeddings[decoded], dtype=np.float32)
    dists = np.linalg.norm(cluster_embs - word_emb[np.newaxis, :], axis=1)

    intra_sens = _filtered_clusant.intra_cluster_sensitivity[selected]
    if intra_sens <= 0:
        intra_sens = 1e-10

    utilities2 = -dists
    log_probs2 = epsilon * utilities2 / (2.0 * intra_sens)
    log_probs2 -= np.max(log_probs2)
    probs2 = np.exp(log_probs2)
    probs2 /= probs2.sum()

    idx = np.random.choice(len(probs2), p=probs2)
    selected_word = _filtered_cluster_word_arrays[selected][idx]
    return _filtered_str_to_tid.get(selected_word)


def sanitize_clusant_filtered(samples, nodes, edges, epsilon,
                               template_target_keys=None,
                               template_name=None, num_clusters=40):
    """CluSanT-F: two-level exponential mechanism over numeric token embeddings.

    Tokenizes each value with RoBERTa, applies CluSanT's two-level mechanism
    in MPNet embedding space per token, restricted to numeric-only tokens.

    Budget: each node gets uniform epsilon, split across BPE tokens
    (eps/n_tokens per token, basic composition).

    Returns:
        (all_values, epsilon_blocks)
    """
    _init_filtered_clusant(epsilon, num_clusters=num_clusters)
    san = Sanitizer()

    epsilon_blocks = []
    all_values = []
    fallback_count = 0
    total_values = 0

    node_decimals = {}
    for name, dec in _TEMPLATE_VALUE_ORDER.get(template_name, []):
        node_decimals[name] = dec

    for si, sample in enumerate(samples):
        topo_order, values = get_topological_order(sample, edges)
        nodes_to_release = [n for n in topo_order if n not in _SKIP_NODES]
        sanitized = {}
        budgets = []

        for node in nodes_to_release:
            total_values += 1
            domain_lower, domain_upper = MEDICAL_DOMAINS.get(
                node, MEDICAL_DOMAINS["_default"])
            true_val = float(values[node])

            # Format value as string
            decimals = node_decimals.get(node, 2)
            if decimals == 0:
                val_str = str(int(true_val))
            else:
                val_str = f"{true_val:.{decimals}f}"

            # Tokenize with RoBERTa
            token_ids = _filtered_roberta_tok.encode(val_str,
                                                      add_special_tokens=False)

            # Structural tokens ("." decimal, "-" sign) are preserved.
            # Only digit tokens consume privacy budget.
            sanitizable = [(i, tid) for i, tid in enumerate(token_ids)
                           if _filtered_roberta_tok.decode([tid]).strip() not in ('.', '-')]
            n_sanitizable = len(sanitizable)
            eps_per_tok = epsilon / max(n_sanitizable, 1)

            # Replace each sanitizable token via CluSanT mechanism
            replacement_ids = list(token_ids)  # start with original
            all_ok = True
            for pos, tid in sanitizable:
                new_tid = _filtered_replace_token(tid, eps_per_tok)
                if new_tid is None:
                    all_ok = False
                    break
                replacement_ids[pos] = new_tid

            if all_ok:
                replacement_str = _filtered_roberta_tok.decode(
                    replacement_ids, skip_special_tokens=True).strip()
                match = re.search(r'-?\d+\.?\d*', replacement_str)
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
            print(f"      [CluSanT-F] {si+1}/{len(samples)} samples done",
                  flush=True)

    if total_values > 0:
        rate = fallback_count / total_values * 100
        print(f"    [CluSanT-F] Fallback rate: {fallback_count}/{total_values} "
              f"({rate:.1f}%)", flush=True)

    return all_values, epsilon_blocks
