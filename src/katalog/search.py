"""Embedding helpers and the public search result type.

Retrieval lives in :mod:`katalog.store`: SQLite stores the vectors and
``sqlite-vec`` computes cosine distance in SQL.  This module only turns text
into vectors and combines the separately embedded product fields.
"""

from __future__ import annotations

import math
import threading
import warnings
from dataclasses import dataclass
from typing import Protocol

from .models import CATEGORY_LABELS, Product

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_RECIPE = "normalized-field-centroid-v1"


@dataclass
class SearchResult:
    product: Product
    score: float
    match_type: str  # 'exact_sku' | 'semantic'
    explanation: str


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedder:
    """Lazy, thread-safe adapter around FastEmbed.

    FastEmbed returns NumPy arrays internally.  Converting them to lists here
    keeps NumPy out of the application's storage and retrieval implementation;
    SQLite receives serialized float32 vectors and performs the ranking.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        from fastembed import TextEmbedding

        with self._lock:
            if self._model is None:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*mean pooling.*")
                    self._model = TextEmbedding(self.model_name)
            return [vector.tolist() for vector in self._model.embed(texts)]


def normalized_embeddings(embedder: Embedder, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return [_unit(vector) for vector in embedder.embed(texts)]


def product_embedding(field_vectors: list[list[float]]) -> list[float]:
    if not field_vectors:
        raise ValueError("a product must have at least one searchable field")
    dimensions = len(field_vectors[0])
    if any(len(vector) != dimensions for vector in field_vectors):
        raise ValueError("embedding dimensions differ")
    centroid = [
        sum(vector[i] for vector in field_vectors) / len(field_vectors)
        for i in range(dimensions)
    ]
    return _unit(centroid)


def searchable_fields(product: Product) -> dict[str, str]:
    fields = {
        "name": product.name,
        "manufacturer": product.manufacturer,
        "category": CATEGORY_LABELS.get(product.category or "", ""),
    }
    if product.enrichment:
        fields["description"] = product.enrichment.description
        fields["applications"] = product.enrichment.applications
    return {name: text for name, text in fields.items() if text}


def _unit(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in vector))
    if length == 0:
        raise ValueError("embedding has zero length")
    return [value / length for value in vector]
