"""Latent-semantic (TF-IDF + truncated SVD) text similarity.

An ALTERNATIVE to the 5-gram MinHash similarity in campaign_similarity.py, not a
replacement: nothing in pipeline.py imports this. MinHash answers "is this the
same form letter, word for word"; this answers "do these two texts say similar
things in similar words", which survives personalisation, reordering and
paraphrase that dilute a 5-gram Jaccard score. It is what campaign_map.py draws.

The pieces, all pure functions over numpy / scipy (no I/O, no plotting):

    X = build_tfidf(texts)                      sparse, L2-normalised rows
    sub = fit_subspace(X, k, weights=...)       truncated SVD of (weighted) X
    Z = project(X, sub)                         comment scores in the subspace
    q = retained_norm(Z)                        how much of each text the subspace kept

Two things matter more than they look:

  * `retained_norm`. X's rows have unit length, so ||Z_i|| is the share of text i
    the k-dimensional subspace retained. A short or idiosyncratic comment can
    keep very little (< 0.1), and its direction is then mostly noise: re-
    normalising it to unit length manufactures a confident-looking cosine: with
    too small a k, campaigns that share no 5-gram at all (even ones on opposite
    sides of the rule) can come out near cosine 1. Gate on `retained_norm`, and
    raise k until such pairs disappear, before trusting a cosine.

  * Fit on DISTINCT texts, weighted by sqrt(copies) (see fit_subspace), so that
    one huge campaign neither owns every axis nor vanishes from the subspace.

Normalisation here is deliberately NOT normalize_campaign_text(): that one DELETES
punctuation, which glues words together across it ("Dear Sir,I" -> "dear siri") and
turns an HTML entity into digits ("I&#39;m" -> "i39m"). Here markup is stripped,
entities unescaped, and punctuation becomes a space, so formatting differences
between copies of one letter do not change its vocabulary.
"""
import collections
import html
import math
import re
from typing import Dict, List, NamedTuple, Optional, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

_TAG_RE = re.compile(r'<[^>]+>')
_NON_ALNUM_RE = re.compile(r'[^a-z0-9]+')

# Function words that carry no campaign signal. Kept short on purpose: TF-IDF
# already downweights common words, this just keeps them out of the bigrams.
STOPWORDS = frozenset(
    'the of and to a in that is for it as with be on this are by or an at from not have has i we our my '
    'you your their they will would should can which was were been but if so than more all any such '
    'these those also its'.split())


def normalize_text(text: Optional[str]) -> str:
    """Lowercase, strip tags, unescape entities, punctuation -> single space."""
    t = html.unescape(_TAG_RE.sub(' ', text or '')).lower()
    return _NON_ALNUM_RE.sub(' ', t).strip()


def text_features(normalized: str) -> List[str]:
    """Unigrams plus bigrams of the non-stopword tokens (bigrams joined by '_')."""
    words = [w for w in normalized.split() if w not in STOPWORDS]
    return words + [a + '_' + b for a, b in zip(words, words[1:])]


class Tfidf(NamedTuple):
    matrix: sp.csr_matrix      # (n_texts, n_terms), rows L2-normalised (all-zero rows stay zero)
    vocab: List[str]           # column index -> term


def build_tfidf(texts: Sequence[str], min_df: int = 2, max_df_frac: float = 0.5) -> Tfidf:
    """Sublinear-tf, smoothed-idf, L2-normalised TF-IDF over word 1-2-grams.

    `texts` are raw; they are normalised here. Terms in fewer than `min_df`
    texts (typos, one-off names) or more than `max_df_frac` of them (docket-wide
    boilerplate) are dropped. A text left with no terms gets an all-zero row, so
    its retained norm is 0 and it is excluded by any retained-norm gate.
    """
    counts = [collections.Counter(text_features(normalize_text(t))) for t in texts]
    n = len(counts)
    df = collections.Counter()
    for c in counts:
        df.update(c.keys())
    vocab = sorted(t for t, d in df.items() if d >= min_df and d <= max_df_frac * n)
    index = {t: i for i, t in enumerate(vocab)}
    idf = np.array([math.log((n + 1) / (df[t] + 1)) + 1.0 for t in vocab]) if vocab else np.zeros(0)

    rows, cols, vals = [], [], []
    for r, c in enumerate(counts):
        for term, f in c.items():
            j = index.get(term)
            if j is not None:
                rows.append(r)
                cols.append(j)
                vals.append((1.0 + math.log(f)) * idf[j])
    X = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)), dtype=np.float64)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return Tfidf(sp.diags(1.0 / norms) @ X, vocab)


class Subspace(NamedTuple):
    components: np.ndarray     # (k, n_terms): right singular vectors, the term LOADINGS of each dimension
    singular_values: np.ndarray
    explained: np.ndarray      # share of the fitted matrix's total variance carried by each dimension


def fit_subspace(X: sp.spmatrix, k: int = 100, weights: Optional[np.ndarray] = None,
                 seed: int = 0) -> Subspace:
    """Truncated SVD of X (optionally row-weighted), uncentred, as in classic LSA.

    `k` is clamped to min(X.shape) - 1 (the most svds can return). `weights`
    multiplies each row before the SVD; pass sqrt(copies) when X holds distinct
    texts. Dimensions come back strongest first. Deterministic for a given seed.
    """
    if X.shape[0] < 2 or X.shape[1] < 2:
        raise ValueError(f'need at least a 2x2 matrix to fit a subspace, got {X.shape}')
    k = max(1, min(k, min(X.shape) - 1))
    Xw = X if weights is None else sp.diags(np.asarray(weights, dtype=float)) @ X
    v0 = np.random.default_rng(seed).standard_normal(min(Xw.shape))
    _, s, vt = svds(Xw, k=k, v0=v0)
    order = np.argsort(-s)
    s, vt = s[order], vt[order]
    total = float(Xw.multiply(Xw).sum()) or 1.0
    return Subspace(vt, s, s ** 2 / total)


def project(X: sp.spmatrix, sub: Subspace) -> np.ndarray:
    """Scores of each row of X in the subspace, shape (n_texts, k)."""
    return np.asarray(X @ sub.components.T)


def retained_norm(Z: np.ndarray) -> np.ndarray:
    """Share of each (unit-length) text the subspace kept; 0 = nothing, 1 = all of it."""
    return np.linalg.norm(Z, axis=1)


def unit_rows(Z: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length (all-zero rows stay zero). Only trust the result
    where retained_norm is high enough - see the module docstring."""
    norm = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(norm, 1e-12)


def group_centroids(Zu: np.ndarray, labels: Sequence,
                    weights: Optional[np.ndarray] = None) -> Dict[object, np.ndarray]:
    """Unit-length (optionally weighted) mean direction of each label's rows of Zu
    (rows already unit length). With distinct texts, weight by copies so the
    centre is the mean over comments, not over distinct wordings."""
    labels = np.asarray(labels, dtype=object)
    w = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=float)
    out = {}
    for lab in dict.fromkeys(labels.tolist()):
        m = (Zu[labels == lab] * w[labels == lab, None]).sum(axis=0)
        norm = np.linalg.norm(m)
        out[lab] = m / norm if norm > 0 else m
    return out


def top_terms(sub: Subspace, vocab: Sequence[str], direction: np.ndarray, n: int = 8) -> List[str]:
    """The n terms that most define `direction` (a unit vector in the subspace),
    i.e. the largest entries of its back-projection into term space."""
    weights = sub.components.T @ direction
    return [vocab[i] for i in np.argsort(-weights)[:n] if weights[i] > 0]
