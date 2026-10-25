"""Embedding models behind a minimal interface.

Retrieval and indexing depend only on ``Embedder``: anything with a stable
``name``, a ``dim`` and an ``encode`` returning L2-normalised float32 rows can
replace the default local Sentence Transformers model.
"""

from __future__ import annotations

import os
from typing import Protocol, Sequence

import numpy as np

from ..config import DEFAULT_MODEL, DEFAULT_MODEL_DIR


class Embedder(Protocol):
    name: str  # part of the cache key: changing it invalidates stored vectors
    dim: int

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray: ...


def normalise_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


class SentenceTransformerEmbedder:
    """Local Sentence Transformers model, loaded lazily on first use.

    After the one-off download the model is loaded with ``local_files_only`` so
    no network request is made at query time.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: str = DEFAULT_MODEL_DIR,
                 device: str = "cpu", allow_download: bool = True):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = device
        self.allow_download = allow_download
        self.name = f"st:{model_name}"
        self.dim = 384 if model_name.endswith("all-MiniLM-L6-v2") else 0  # confirmed on load
        self._model = None
        self.load_seconds: float | None = None

    def _load(self):
        if self._model is None:
            import time

            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
            from sentence_transformers import SentenceTransformer

            start = time.perf_counter()
            try:
                model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir, device=self.device,
                                            local_files_only=True)
            except Exception:
                if not self.allow_download:
                    raise
                model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir, device=self.device)
            self.load_seconds = time.perf_counter() - start
            get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
            self.dim = int(get_dim())
            self._model = model
        return self._model

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 384), dtype=np.float32)
        model = self._load()
        vectors = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
        return normalise_rows(vectors)


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one normalised query against normalised rows."""
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    return matrix @ query.reshape(-1).astype(np.float32)
