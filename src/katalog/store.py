"""Persistent catalogue and vector search backed by SQLite.

The public interface is deliberately small: seed once, add/delete products at
runtime, and search.  Relational writes, aliases, vector serialization and
cosine ranking are one atomic implementation detail behind that interface.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import sqlite_vec

from .models import (
    CATEGORY_LABELS,
    Attributes,
    Catalogue,
    Enrichment,
    Issue,
    Package,
    Price,
    Product,
)
from .normalize import (
    check_package_plausibility,
    fold,
    normalize_sku,
    normalize_text,
    parse_attributes,
    parse_package,
    parse_price,
)
from .search import (
    EMBEDDING_RECIPE,
    Embedder,
    FastEmbedder,
    SearchResult,
    normalized_embeddings,
    product_embedding,
    searchable_fields,
)

SCHEMA_VERSION = "1"
SCHEMA_ID = "catalogue-sqlite-vec-v1"


class ProductConflict(ValueError):
    pass


class ProductNotFound(LookupError):
    pass


class InvalidProduct(ValueError):
    pass


@dataclass(frozen=True)
class ProductInput:
    sku: str
    name: str
    manufacturer: str
    category: str | None = None
    package: str = ""
    price: str = ""
    attributes: str = ""
    description: str = ""
    applications: str = ""


@dataclass(frozen=True)
class DeletedProduct:
    canonical_sku: str
    aliases: list[str]


SCHEMA = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) STRICT;

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    canonical_sku TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    manufacturer TEXT NOT NULL,
    manufacturer_folded TEXT NOT NULL,
    category TEXT,
    payload_json TEXT NOT NULL,
    source_kind TEXT NOT NULL CHECK (source_kind IN ('seed', 'runtime')),
    created_at TEXT NOT NULL
) STRICT;

CREATE TABLE product_identifiers (
    sku TEXT PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('canonical', 'alias'))
) STRICT;

CREATE TABLE product_embeddings (
    product_id INTEGER PRIMARY KEY REFERENCES products(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL
) STRICT;

CREATE TABLE field_embeddings (
    product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    field_name TEXT NOT NULL,
    source_text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (product_id, field_name)
) STRICT;

CREATE TABLE source_records (
    row_number INTEGER PRIMARY KEY,
    payload_json TEXT NOT NULL
) STRICT;

CREATE TABLE external_candidates (
    sku TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL
) STRICT;

CREATE TABLE catalogue_events (
    id INTEGER PRIMARY KEY,
    action TEXT NOT NULL CHECK (action IN ('seed', 'add', 'delete')),
    canonical_sku TEXT,
    occurred_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
) STRICT;

CREATE INDEX products_by_category ON products(category);
CREATE INDEX products_by_manufacturer ON products(manufacturer_folded);
"""


class CatalogueStore:
    def __init__(self, path: Path, embedder: Embedder | None = None):
        self.path = path
        self.embedder = embedder or FastEmbedder()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("BEGIN EXCLUSIVE")
            tables = {
                row["name"]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if not tables:
                for statement in SCHEMA.split(";"):
                    if statement.strip():
                        db.execute(statement)
                db.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    [
                        ("schema_version", SCHEMA_VERSION),
                        ("schema_id", SCHEMA_ID),
                        ("embedding_model", self.embedder.model_name),
                        ("embedding_recipe", EMBEDDING_RECIPE),
                    ],
                )
            elif "metadata" not in tables:
                raise RuntimeError("existing database has no supported schema metadata")
            version = self._metadata(db, "schema_version")
            if version != SCHEMA_VERSION:
                raise RuntimeError(f"unsupported database schema {version!r}")
            if self._metadata(db, "schema_id") != SCHEMA_ID:
                raise RuntimeError(
                    "existing database does not match the supported schema"
                )
            stored_model = self._metadata(db, "embedding_model")
            if stored_model != self.embedder.model_name:
                raise RuntimeError(
                    f"database uses embedding model {stored_model!r}, "
                    f"configured model is {self.embedder.model_name!r}"
                )
            stored_recipe = self._metadata(db, "embedding_recipe")
            if stored_recipe != EMBEDDING_RECIPE:
                raise RuntimeError(
                    f"database uses embedding recipe {stored_recipe!r}, "
                    f"configured recipe is {EMBEDDING_RECIPE!r}"
                )

    @property
    def seeded(self) -> bool:
        with self._connect() as db:
            return self._metadata(db, "initial_seed_completed") == "1"

    def seed(self, catalogue: Catalogue) -> None:
        """Import the normalized sample exactly once and persist its vectors."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if self._metadata(db, "initial_seed_completed") == "1":
                return
            prepared = self._prepare_products(catalogue.products)
            dimensions = len(prepared[0][1]) if prepared else None
            now = _now()
            self._record_dimensions(db, dimensions)

            for product, aggregate, fields in prepared:
                self._insert_product(db, product, aggregate, fields, "seed", now)
            db.executemany(
                "INSERT INTO source_records(row_number, payload_json) VALUES (?, ?)",
                [
                    (record.raw.row_number, _json(record))
                    for record in catalogue.records
                ],
            )
            db.executemany(
                "INSERT INTO external_candidates(sku, payload_json) VALUES (?, ?)",
                [
                    (candidate.sku, _json(candidate))
                    for candidate in catalogue.external_candidates
                ],
            )
            db.execute(
                "INSERT INTO catalogue_events(action, occurred_at, payload_json) VALUES ('seed', ?, ?)",
                (now, json.dumps({"products": len(catalogue.products)})),
            )
            db.execute(
                "INSERT INTO metadata(key, value) VALUES ('initial_seed_completed', '1')"
            )

    def add(self, data: ProductInput) -> Product:
        product = self._runtime_product(data)
        aggregate, fields = self._prepare_product(product)
        now = _now()

        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            collision = db.execute(
                "SELECT sku FROM product_identifiers WHERE sku = ?",
                (product.canonical_sku,),
            ).fetchone()
            if collision:
                raise ProductConflict(f"SKU {product.canonical_sku} already exists")
            self._record_dimensions(db, len(aggregate))
            self._insert_product(db, product, aggregate, fields, "runtime", now)
            db.execute(
                """INSERT INTO catalogue_events(action, canonical_sku, occurred_at, payload_json)
                   VALUES ('add', ?, ?, ?)""",
                (product.canonical_sku, now, _json(_product_payload(product))),
            )
        return product

    def delete(self, sku: str) -> DeletedProduct:
        normalized = normalize_sku(sku)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """SELECT p.id, p.canonical_sku, p.payload_json
                   FROM product_identifiers i
                   JOIN products p ON p.id = i.product_id
                   WHERE i.sku = ?""",
                (normalized,),
            ).fetchone()
            if row is None:
                raise ProductNotFound(f"SKU {normalized} does not exist")
            aliases = [
                item["sku"]
                for item in db.execute(
                    "SELECT sku FROM product_identifiers WHERE product_id = ? AND kind = 'alias' ORDER BY sku",
                    (row["id"],),
                )
            ]
            deleted = DeletedProduct(row["canonical_sku"], aliases)
            audit_payload = {
                "deleted": asdict(deleted),
                "product": json.loads(row["payload_json"]),
            }
            db.execute("DELETE FROM products WHERE id = ?", (row["id"],))
            db.execute(
                """INSERT INTO catalogue_events(action, canonical_sku, occurred_at, payload_json)
                   VALUES ('delete', ?, ?, ?)""",
                (deleted.canonical_sku, _now(), _json(audit_payload)),
            )
            return deleted

    def search(
        self,
        query: str,
        *,
        manufacturer: str | None = None,
        category: str | None = None,
        limit: int = 5,
    ) -> list[SearchResult]:
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        manufacturer = normalize_text(manufacturer or "") or None
        category = normalize_text(category or "") or None
        with self._connect() as db:
            db.execute("BEGIN")
            exact = self._exact(db, query)
            if exact is not None:
                return [exact] if _passes(exact.product, manufacturer, category) else []
            return self._semantic(db, query, manufacturer, category, limit)

    def count_products(self) -> int:
        with self._connect() as db:
            return int(db.execute("SELECT count(*) FROM products").fetchone()[0])

    def _exact(
        self,
        db: sqlite3.Connection,
        query: str,
    ) -> SearchResult | None:
        sku = normalize_sku(query)
        row = db.execute(
            """SELECT p.payload_json
                FROM product_identifiers i
                JOIN products p ON p.id = i.product_id
                WHERE i.sku = ?""",
            (sku,),
        ).fetchone()
        if row is None:
            return None
        product = _product_from_payload(json.loads(row["payload_json"]))
        if sku == product.canonical_sku:
            reason = f"exact match on catalogue number {sku}"
        else:
            reason = f"{sku} is a registered alias of {product.canonical_sku}"
        return SearchResult(product, 1.0, "exact_sku", f"score 1.00 - {reason}")

    def _semantic(
        self,
        db: sqlite3.Connection,
        query: str,
        manufacturer: str | None,
        category: str | None,
        limit: int,
    ) -> list[SearchResult]:
        query_vector = normalized_embeddings(self.embedder, [query])[0]
        self._validate_dimensions(db, len(query_vector))
        query_blob = sqlite_vec.serialize_float32(query_vector)
        clauses = ["1 = 1"]
        values: list[object] = [query_blob]
        _add_filters(clauses, values, manufacturer, category)
        values.append(limit)
        rows = db.execute(
            f"""SELECT p.id, p.payload_json,
                       1.0 - vec_distance_cosine(pe.embedding, ?) AS score
                FROM products p
                JOIN product_embeddings pe ON pe.product_id = p.id
                WHERE {" AND ".join(clauses)}
                ORDER BY score DESC, p.canonical_sku
                LIMIT ?""",
            values,
        ).fetchall()
        if not rows:
            return []

        ids = [row["id"] for row in rows]
        placeholders = ",".join("?" for _ in ids)
        field_scores: dict[int, list[tuple[str, float]]] = {
            product_id: [] for product_id in ids
        }
        for field in db.execute(
            f"""SELECT product_id, field_name,
                       1.0 - vec_distance_cosine(embedding, ?) AS score
                FROM field_embeddings
                WHERE product_id IN ({placeholders})""",
            [query_blob, *ids],
        ):
            field_scores[field["product_id"]].append(
                (field["field_name"], float(field["score"]))
            )

        results = []
        for row in rows:
            product = _product_from_payload(json.loads(row["payload_json"]))
            score = float(row["score"])
            ranked = sorted(
                field_scores[row["id"]], key=lambda item: (-item[1], item[0])
            )[:2]
            explanation = self._explanation(
                product, score, ranked, manufacturer, category
            )
            results.append(SearchResult(product, score, "semantic", explanation))
        return results

    def _explanation(
        self,
        product: Product,
        score: float,
        fields: list[tuple[str, float]],
        manufacturer: str | None,
        category: str | None,
    ) -> str:
        drivers = ", ".join(f"{name} {value:.2f}" for name, value in fields)
        parts = [f"score {score:.2f} (cosine similarity, not a probability)"]
        parts.append(f"strongest fields: {drivers}")
        if product.category:
            source = (
                " (inferred)"
                if product.category_source == "inferred_from_product_name"
                else ""
            )
            parts.append(f"category {product.category}{source}")
        if product.enrichment:
            origin = (
                "description added with the product"
                if product.enrichment.match_rule == "runtime_input"
                else "description harvested from the manufacturer's page"
            )
            parts.append(origin)
        if manufacturer or category:
            applied = ", ".join(value for value in (manufacturer, category) if value)
            parts.append(f"passed filter: {applied}")
        return " - ".join(parts)

    def _runtime_product(self, data: ProductInput) -> Product:
        sku = normalize_sku(data.sku)
        name = normalize_text(data.name)
        manufacturer = normalize_text(data.manufacturer)
        if not sku or not name or not manufacturer:
            raise InvalidProduct("sku, name and manufacturer must not be empty")
        category = data.category.strip().upper() if data.category else None
        if category is not None and category not in CATEGORY_LABELS:
            raise InvalidProduct(f"unknown category {data.category!r}")

        package = parse_package(data.package)
        price = parse_price(data.price)
        attributes = parse_attributes(data.attributes)
        issues = []
        if package.unit == "unknown":
            issues.append(
                Issue(
                    "package_unit_unknown", f"cannot read a unit from {data.package!r}"
                )
            )
        if price.amount is None:
            issues.append(Issue("price_missing", f"no price in {data.price!r}"))
        issues.extend(check_package_plausibility(category, package))
        enrichment = None
        if data.description.strip() or data.applications.strip():
            enrichment = Enrichment(
                source_sku=sku,
                name=name,
                description=normalize_text(data.description),
                applications=normalize_text(data.applications),
                match_rule="runtime_input",
            )
        return Product(
            canonical_sku=sku,
            name=name,
            manufacturer=manufacturer,
            category=category,
            category_source="runtime",
            package=package,
            price=price,
            attributes=attributes,
            issues=issues,
            enrichment=enrichment,
        )

    def _prepare_products(
        self, products: list[Product]
    ) -> list[tuple[Product, list[float], dict[str, tuple[str, list[float]]]]]:
        searchable = [searchable_fields(product) for product in products]
        texts = [text for fields in searchable for text in fields.values()]
        vectors = normalized_embeddings(self.embedder, texts)
        prepared = []
        cursor = 0
        for product, fields in zip(products, searchable):
            count = len(fields)
            product_vectors = vectors[cursor : cursor + count]
            cursor += count
            named = {
                name: (text, vector)
                for (name, text), vector in zip(fields.items(), product_vectors)
            }
            prepared.append((product, product_embedding(product_vectors), named))
        return prepared

    def _prepare_product(
        self, product: Product
    ) -> tuple[list[float], dict[str, tuple[str, list[float]]]]:
        _, aggregate, fields = self._prepare_products([product])[0]
        return aggregate, fields

    def _insert_product(
        self,
        db: sqlite3.Connection,
        product: Product,
        aggregate: list[float],
        fields: dict[str, tuple[str, list[float]]],
        source_kind: str,
        created_at: str,
    ) -> None:
        cursor = db.execute(
            """INSERT INTO products(
                   canonical_sku, name, manufacturer, manufacturer_folded,
                   category, payload_json, source_kind, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                product.canonical_sku,
                product.name,
                product.manufacturer,
                fold(product.manufacturer),
                product.category,
                _json(_product_payload(product)),
                source_kind,
                created_at,
            ),
        )
        product_id = int(cursor.lastrowid)
        identifiers = [(product.canonical_sku, product_id, "canonical")]
        identifiers.extend((alias, product_id, "alias") for alias in product.alias_skus)
        db.executemany(
            "INSERT INTO product_identifiers(sku, product_id, kind) VALUES (?, ?, ?)",
            identifiers,
        )
        db.execute(
            "INSERT INTO product_embeddings(product_id, embedding) VALUES (?, ?)",
            (product_id, sqlite_vec.serialize_float32(aggregate)),
        )
        db.executemany(
            """INSERT INTO field_embeddings(product_id, field_name, source_text, embedding)
               VALUES (?, ?, ?, ?)""",
            [
                (product_id, name, text, sqlite_vec.serialize_float32(vector))
                for name, (text, vector) in fields.items()
            ],
        )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.enable_load_extension(True)
        try:
            sqlite_vec.load(db)
        finally:
            db.enable_load_extension(False)
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("PRAGMA busy_timeout = 5000")
        return db

    @staticmethod
    def _metadata(db: sqlite3.Connection, key: str) -> str | None:
        row = db.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _record_dimensions(
        self, db: sqlite3.Connection, dimensions: int | None
    ) -> None:
        if dimensions is None:
            return
        stored = self._metadata(db, "embedding_dimensions")
        if stored is None:
            db.execute(
                "INSERT INTO metadata(key, value) VALUES ('embedding_dimensions', ?)",
                (str(dimensions),),
            )
        elif int(stored) != dimensions:
            raise RuntimeError(
                f"database uses {stored}-dimensional embeddings, got {dimensions} dimensions"
            )

    def _validate_dimensions(self, db: sqlite3.Connection, dimensions: int) -> None:
        stored = self._metadata(db, "embedding_dimensions")
        if stored is not None and int(stored) != dimensions:
            raise RuntimeError(
                f"database uses {stored}-dimensional embeddings, got {dimensions} dimensions"
            )


def _add_filters(
    clauses: list[str],
    values: list[object],
    manufacturer: str | None,
    category: str | None,
) -> None:
    if manufacturer:
        clauses.append("instr(p.manufacturer_folded, ?) > 0")
        values.append(fold(manufacturer.strip()))
    if category:
        clauses.append("p.category = ?")
        values.append(category.strip().upper())


def _passes(product: Product, manufacturer: str | None, category: str | None) -> bool:
    if manufacturer and fold(manufacturer) not in fold(product.manufacturer):
        return False
    return (
        not category
        or (product.category or "").casefold() == category.strip().casefold()
    )


def _product_payload(product: Product) -> dict:
    return {
        "canonical_sku": product.canonical_sku,
        "name": product.name,
        "manufacturer": product.manufacturer,
        "category": product.category,
        "category_source": product.category_source,
        "package": {
            "quantity": str(product.package.quantity)
            if product.package.quantity is not None
            else None,
            "unit": product.package.unit,
            "raw": product.package.raw,
        },
        "price": {
            "amount": str(product.price.amount)
            if product.price.amount is not None
            else None,
            "currency": product.price.currency,
            "currency_inferred": product.price.currency_inferred,
            "raw": product.price.raw,
        },
        "attributes": asdict(product.attributes),
        "source_rows": product.source_rows,
        "alias_skus": product.alias_skus,
        "resolution_notes": product.resolution_notes,
        "issues": [asdict(issue) for issue in product.issues],
        "enrichment": asdict(product.enrichment) if product.enrichment else None,
    }


def _product_from_payload(payload: dict) -> Product:
    package = payload["package"]
    price = payload["price"]
    enrichment = payload["enrichment"]
    return Product(
        canonical_sku=payload["canonical_sku"],
        name=payload["name"],
        manufacturer=payload["manufacturer"],
        category=payload["category"],
        category_source=payload["category_source"],
        package=Package(
            Decimal(package["quantity"]) if package["quantity"] is not None else None,
            package["unit"],
            package["raw"],
        ),
        price=Price(
            Decimal(price["amount"]) if price["amount"] is not None else None,
            price["currency"],
            price["currency_inferred"],
            price["raw"],
        ),
        attributes=Attributes(**payload["attributes"]),
        source_rows=payload["source_rows"],
        alias_skus=payload["alias_skus"],
        resolution_notes=payload["resolution_notes"],
        issues=[Issue(**issue) for issue in payload["issues"]],
        enrichment=Enrichment(**enrichment) if enrichment else None,
    )


def _json(value) -> str:
    def default(item):
        if isinstance(item, Decimal):
            return str(item)
        if is_dataclass(item):
            return asdict(item)
        raise TypeError(f"cannot encode {type(item).__name__}")

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=default)


def _now() -> str:
    return datetime.now(UTC).isoformat()
