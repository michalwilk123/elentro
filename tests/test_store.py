from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from katalog.models import Catalogue
from katalog.pipeline import build_catalogue
from katalog.store import (
    CatalogueStore,
    ProductConflict,
    ProductInput,
    ProductNotFound,
)


class FakeEmbedder:
    model_name = "fake-embedding-v1"

    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        folded = text.casefold()
        return [
            1.0,
            3.0 if "zimn" in folded or "cold" in folded else 0.1,
            3.0 if "prob" in folded or "tube" in folded else 0.1,
            3.0 if "waga" in folded or "balance" in folded else 0.1,
        ]


@pytest.fixture
def store(tmp_path):
    embedder = FakeEmbedder()
    result = CatalogueStore(tmp_path / "catalogue.sqlite3", embedder)
    result.seed(Catalogue(products=[], external_candidates=[], records=[]))
    return result


def test_runtime_add_is_immediately_searchable_and_persists(store):
    store.add(
        ProductInput(
            sku="AC-10001",
            name="Probówki do zimnych próbek",
            manufacturer="ACME Labs",
            category="LAB_PLASTICWARE",
            package="50 szt.",
            price="99 PLN",
            description="Cold sample tubes",
        )
    )

    exact = store.search("ac-10001")
    assert exact[0].match_type == "exact_sku"
    semantic = store.search(
        "cold tubes", manufacturer="acme", category="LAB_PLASTICWARE"
    )
    assert semantic[0].product.canonical_sku == "AC-10001"
    assert "description added with the product" in semantic[0].explanation

    reopened_embedder = FakeEmbedder()
    reopened = CatalogueStore(store.path, reopened_embedder)
    assert reopened.search("AC-10001")[0].product.name == "Probówki do zimnych próbek"
    assert (
        reopened_embedder.calls == []
    )  # exact lookup does not recompute stored vectors


def test_runtime_delete_by_sku_removes_relational_and_vector_rows(store):
    store.add(ProductInput("AC-10001", "Cold tube", "ACME", "LAB_PLASTICWARE"))
    deleted = store.delete("ac-10001")
    assert deleted.canonical_sku == "AC-10001"
    assert store.search("AC-10001") == []
    assert store.count_products() == 0

    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT count(*) FROM product_embeddings").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM field_embeddings").fetchone()[0] == 0
        assert (
            db.execute(
                "SELECT action FROM catalogue_events ORDER BY id DESC"
            ).fetchone()[0]
            == "delete"
        )

    with pytest.raises(ProductNotFound):
        store.delete("AC-10001")

    assert CatalogueStore(store.path, FakeEmbedder()).count_products() == 0


def test_alias_collision_and_delete_resolve_to_the_canonical_product(tmp_path):
    store = CatalogueStore(tmp_path / "aliases.sqlite3", FakeEmbedder())
    store.seed(build_catalogue())

    with pytest.raises(ProductConflict):
        store.add(ProductInput("CH-10248A", "Duplicate alias", "ACME"))

    deleted = store.delete("ch-10248a")
    assert deleted.canonical_sku == "CH-10248"
    assert deleted.aliases == ["CH-10248A"]
    with pytest.raises(ProductNotFound):
        store.delete("CH-10248")

    with sqlite3.connect(store.path) as db:
        payload = db.execute(
            "SELECT payload_json FROM catalogue_events WHERE action = 'delete' ORDER BY id DESC"
        ).fetchone()[0]
    assert '"source_rows":[36,222]' in payload
    assert '"alias_skus":["CH-10248A"]' in payload


def test_duplicate_runtime_sku_is_rejected_without_partial_rows(store):
    product = ProductInput("AC-10001", "Cold tube", "ACME")
    store.add(product)
    with pytest.raises(ProductConflict):
        store.add(product)
    assert store.count_products() == 1


def test_embedding_failure_does_not_leave_a_partial_product(tmp_path):
    class BrokenEmbedder(FakeEmbedder):
        def embed(self, texts):
            raise RuntimeError("model failed")

    store = CatalogueStore(tmp_path / "broken.sqlite3", BrokenEmbedder())
    store.seed(Catalogue(products=[], external_candidates=[], records=[]))
    with pytest.raises(RuntimeError, match="model failed"):
        store.add(ProductInput("AC-10001", "Cold tube", "ACME"))
    assert store.count_products() == 0


def test_empty_catalogue_is_not_reseeded_after_deleting_everything(store):
    store.add(ProductInput("AC-10001", "Cold tube", "ACME"))
    store.delete("AC-10001")
    reopened = CatalogueStore(store.path, FakeEmbedder())
    assert reopened.seeded is True
    reopened.seed(Catalogue(products=[], external_candidates=[], records=[]))
    assert reopened.count_products() == 0


def test_database_rejects_a_different_embedding_model(store):
    class OtherEmbedder(FakeEmbedder):
        model_name = "different-model"

    with pytest.raises(RuntimeError, match="database uses embedding model"):
        CatalogueStore(store.path, OtherEmbedder())


def test_add_before_seed_records_and_validates_embedding_dimensions(tmp_path):
    path = tmp_path / "unseeded.sqlite3"
    store = CatalogueStore(path, FakeEmbedder())
    store.add(ProductInput("AC-1", "Cold tube", "ACME"))

    class WrongDimensions(FakeEmbedder):
        def embed(self, texts):
            return [[1.0, 2.0] for _ in texts]

    reopened = CatalogueStore(path, WrongDimensions())
    with pytest.raises(RuntimeError, match="dimensional embeddings"):
        reopened.search("cold")
    with pytest.raises(RuntimeError, match="dimensional embeddings"):
        reopened.add(ProductInput("AC-2", "Other", "ACME"))


def test_unversioned_existing_database_is_rejected(tmp_path):
    path = tmp_path / "partial.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE products(id INTEGER PRIMARY KEY)")
    with pytest.raises(RuntimeError, match="no supported schema metadata"):
        CatalogueStore(path, FakeEmbedder())


def test_whitespace_filter_is_ignored_for_exact_and_semantic_paths(store):
    store.add(ProductInput("AC-1", "Cold tube", "ACME"))
    assert store.search("AC-1", manufacturer=" ")[0].match_type == "exact_sku"
    assert store.search("cold", manufacturer=" ")[0].match_type == "semantic"


def test_concurrent_seed_is_idempotent(tmp_path):
    path = tmp_path / "concurrent.sqlite3"
    stores = [
        CatalogueStore(path, FakeEmbedder()),
        CatalogueStore(path, FakeEmbedder()),
    ]
    catalogue = Catalogue(products=[], external_candidates=[], records=[])
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda store: store.seed(catalogue), stores))

    with sqlite3.connect(path) as db:
        assert (
            db.execute(
                "SELECT count(*) FROM catalogue_events WHERE action = 'seed'"
            ).fetchone()[0]
            == 1
        )
