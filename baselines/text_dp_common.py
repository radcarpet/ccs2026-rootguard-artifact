"""
Common infrastructure for text-based DP sanitization baselines.

Provides:
  - RoBERTa tokenizer and static embedding matrix (cached, lazy-loaded)
  - Word-level tokenization of medical text
  - Euclidean distance computation between token embeddings
  - Numeric value extraction from sanitized text with fallback

Used by: SanText, CusText, CluSanT, CAPE baselines.
"""

import re
import numpy as np

# Lazy-loaded globals
_tokenizer = None
_embedding_matrix = None  # shape: (vocab_size, embed_dim), numpy float32
_vocab_size = None


def _load_model():
    """Lazy-load RoBERTa tokenizer and embedding matrix."""
    global _tokenizer, _embedding_matrix, _vocab_size
    if _tokenizer is not None:
        return

    import torch
    from transformers import AutoTokenizer, AutoModel

    model_name = "roberta-base"
    _tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name)
    # Extract the static word embedding matrix (before positional/layer-norm)
    _embedding_matrix = model.embeddings.word_embeddings.weight.detach().cpu().numpy()
    _vocab_size = _embedding_matrix.shape[0]

    # Free the full model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_tokenizer():
    _load_model()
    return _tokenizer


def get_embedding_matrix():
    _load_model()
    return _embedding_matrix


def get_vocab_size():
    _load_model()
    return _vocab_size


def tokenize_text(text):
    """Tokenize text using the RoBERTa tokenizer, return token IDs."""
    _load_model()
    encoded = _tokenizer.encode(text, add_special_tokens=False)
    return encoded


def decode_tokens(token_ids):
    """Decode a list of token IDs back to text."""
    _load_model()
    return _tokenizer.decode(token_ids, skip_special_tokens=True)


def get_token_embedding(token_id):
    """Get the embedding vector for a single token ID."""
    _load_model()
    return _embedding_matrix[token_id]


def euclidean_distances_from(token_id):
    """Compute Euclidean distances from one token to all vocabulary tokens.

    Returns:
        1D numpy array of shape (vocab_size,) with distances.
    """
    _load_model()
    emb = _embedding_matrix[token_id]  # (embed_dim,)
    # Efficient: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    diff = _embedding_matrix - emb[np.newaxis, :]
    dists = np.linalg.norm(diff, axis=1)
    return dists


def cosine_distances_from(token_id):
    """Compute cosine distances (1 - cosine_similarity) from one token to all.

    Returns:
        1D numpy array of shape (vocab_size,) with distances in [0, 2].
    """
    _load_model()
    emb = _embedding_matrix[token_id]
    emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
    norms = np.linalg.norm(_embedding_matrix, axis=1, keepdims=True) + 1e-10
    normed = _embedding_matrix / norms
    cosine_sim = normed @ emb_norm
    return 1.0 - cosine_sim


# ---------------------------------------------------------------------------
# Numeric token identification
# ---------------------------------------------------------------------------

# Must contain at least one digit — excludes pure dash/dot sequences
# ("---", "...", etc.) that RoBERTa BPE includes as formatting tokens.
_NUMERIC_CHAR_PATTERN = re.compile(r'^[\d.\-]*\d[\d.\-]*$')


def identify_numeric_tokens(token_ids):
    """Identify which token positions contain numeric characters.

    A token is considered numeric if its decoded text (stripped) consists
    entirely of digits, decimal points, or minus signs.

    Args:
        token_ids: List of token IDs

    Returns:
        List of booleans, True for numeric token positions.
    """
    _load_model()
    is_numeric = []
    for tid in token_ids:
        decoded = _tokenizer.decode([tid]).strip()
        is_numeric.append(bool(decoded and _NUMERIC_CHAR_PATTERN.match(decoded)))
    return is_numeric


# ---------------------------------------------------------------------------
# Numeric extraction helpers
# ---------------------------------------------------------------------------

_NUMBER_PATTERN = re.compile(r'-?\d+\.?\d*')


# ---------------------------------------------------------------------------
# RoBERTa numeric vocabulary mask (shared across -F baselines)
# ---------------------------------------------------------------------------

_roberta_numeric_mask = None   # (vocab_size,) boolean array
_roberta_numeric_ids = None    # int array of numeric token IDs


def build_roberta_numeric_mask():
    """Build a boolean mask over RoBERTa's vocabulary identifying numeric tokens.

    A token is considered numeric if its decoded text (stripped of whitespace)
    consists entirely of digits, decimal points, or minus signs.

    Returns:
        numpy array of shape (vocab_size,) with True for numeric tokens.
    """
    global _roberta_numeric_mask
    if _roberta_numeric_mask is not None:
        return _roberta_numeric_mask

    _load_model()
    _roberta_numeric_mask = np.zeros(_vocab_size, dtype=bool)
    for idx in range(_vocab_size):
        decoded = _tokenizer.decode([idx]).strip()
        if decoded and _NUMERIC_CHAR_PATTERN.match(decoded):
            _roberta_numeric_mask[idx] = True

    n = int(_roberta_numeric_mask.sum())
    print(f"    [text_dp_common] RoBERTa numeric mask: {n}/{_vocab_size} tokens",
          flush=True)
    return _roberta_numeric_mask


def get_numeric_token_ids():
    """Return integer array of all RoBERTa token IDs that decode to numeric strings.

    These are the candidate tokens for all -F (filtered) baselines.
    """
    global _roberta_numeric_ids
    if _roberta_numeric_ids is not None:
        return _roberta_numeric_ids

    mask = build_roberta_numeric_mask()
    _roberta_numeric_ids = np.where(mask)[0]
    return _roberta_numeric_ids


def extract_numeric_values_from_text(text, node_order):
    """Extract numeric values from sanitized text in the order they appear.

    Args:
        text: The sanitized text string
        node_order: List of (node_name, decimal_places) in expected order

    Returns:
        Dict {node_name: float_value} for successfully parsed values,
        and set of node_names that failed to parse.
    """
    numbers = _NUMBER_PATTERN.findall(text)
    parsed = {}
    failed = set()

    for i, (node_name, _) in enumerate(node_order):
        if i < len(numbers):
            try:
                parsed[node_name] = float(numbers[i])
            except ValueError:
                failed.add(node_name)
        else:
            failed.add(node_name)

    return parsed, failed
