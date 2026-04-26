"""Lightweight semantic-similarity utilities used to merge clusters.

Two backends, picked automatically:

  Tier 1 — TF-IDF (default).  ``scikit-learn`` is a small, well-tested
           dependency that ships pre-built wheels for every OS we
           target.  Tokenisation reuses ``folderangel.morph`` so
           Korean compound nouns + Latin brand tokens are first-class.
           Deterministic, fast, no model download.

  Tier 2 — Sentence-transformer embeddings (``sentence_transformers``
           extra).  Only activates when the user installed the
           ``embed`` extra and a small multilingual model is loadable.
           Higher recall on truly distinct filenames whose bodies
           talk about the same project.

The public API hides the choice — callers ask for a similarity matrix
or pairwise similarity and the right backend runs.

If neither backend is available (no sklearn, no torch+ST), every
public function returns a no-op signal so the planner falls back to
pure-signature clustering.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import morph

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

_TFIDF_VECTORIZER = None
_TFIDF_AVAILABLE: Optional[bool] = None
_ST_MODEL = None
_ST_AVAILABLE: Optional[bool] = None
_PREFERRED_ST_MODELS = (
    "BAAI/bge-m3",
    "BAAI/bge-small-en-v1.5",
    "intfloat/multilingual-e5-small",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)


def _korean_tokenizer(text: str) -> list[str]:
    """Tokeniser for sklearn's ``TfidfVectorizer`` — runs through the
    same morpheme analyser the signature builder uses, so the term
    space matches what we already trust."""
    return morph.extract_nouns(text or "")


def _have_sklearn() -> bool:
    global _TFIDF_AVAILABLE
    if _TFIDF_AVAILABLE is not None:
        return _TFIDF_AVAILABLE
    try:
        import sklearn  # noqa: F401

        _TFIDF_AVAILABLE = True
    except Exception:
        _TFIDF_AVAILABLE = False
    return _TFIDF_AVAILABLE


def _have_st() -> bool:
    global _ST_AVAILABLE
    if _ST_AVAILABLE is not None:
        return _ST_AVAILABLE
    try:
        import sentence_transformers  # noqa: F401

        _ST_AVAILABLE = True
    except Exception:
        _ST_AVAILABLE = False
    return _ST_AVAILABLE


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _load_st_model():
    global _ST_MODEL
    if _ST_MODEL is not None:
        return _ST_MODEL
    if not _have_st():
        return None
    from sentence_transformers import SentenceTransformer  # type: ignore

    for name in _PREFERRED_ST_MODELS:
        try:
            _ST_MODEL = SentenceTransformer(name)
            log.info("loaded sentence-transformer model: %s", name)
            return _ST_MODEL
        except Exception as exc:
            log.debug("ST model %s not loadable: %s", name, exc)
    return None


def backend_label() -> str:
    """Which backend will run, for the progress log."""
    if _ST_MODEL is not None:
        return "embedding"
    if _have_st() and _load_st_model() is not None:
        return "embedding"
    if _have_sklearn():
        return "tfidf"
    return "none"


def embed(texts: list[str]) -> Optional[np.ndarray]:
    """Compute an L2-normalised similarity space for *texts*.

    Returns ``None`` when no backend is available (caller treats it
    as "skip the merge step").  Otherwise returns a 2-D float array
    of shape ``(len(texts), D)``.
    """
    if not texts:
        return None
    # Try ST first if explicitly available.
    model = _load_st_model()
    if model is not None:
        try:
            vecs = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return np.asarray(vecs, dtype=np.float32)
        except Exception as exc:
            log.warning("ST encode failed (%s) — falling back to tfidf", exc)
    # TF-IDF fallback.
    if _have_sklearn():
        from sklearn.feature_extraction.text import TfidfVectorizer

        vec = TfidfVectorizer(
            tokenizer=_korean_tokenizer,
            token_pattern=None,        # we provide our own tokens
            lowercase=False,           # tokeniser already normalises
            min_df=1,
            ngram_range=(1, 1),
            norm="l2",
        )
        try:
            mat = vec.fit_transform(texts)
        except ValueError:
            # All documents empty or only stop tokens.  No signal.
            return None
        # Sparse → dense float32 (small dim per document for our use)
        return mat.astype(np.float32).toarray()
    return None


# ---------------------------------------------------------------------------
# Cluster merging
# ---------------------------------------------------------------------------

def _pairwise_cosine(vecs: np.ndarray) -> np.ndarray:
    """Cosine similarity for L2-normalised vectors == dot product."""
    # Defensive — re-normalise in case the backend skipped it.
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    v = vecs / norms
    return v @ v.T


def merge_by_similarity(
    docs: list[str],
    *,
    threshold: float = 0.55,
) -> list[list[int]]:
    """Greedy single-link merge: return groups of indices whose
    pairwise similarity is ≥ ``threshold``.

    Returns ``[[0], [1], …]`` (no merging) when no backend is
    available — the caller's existing clustering then stays as-is.
    """
    n = len(docs)
    if n == 0:
        return []
    vecs = embed(docs)
    if vecs is None:
        return [[i] for i in range(n)]

    sims = _pairwise_cosine(vecs)
    # Union-Find merge by threshold.
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())
