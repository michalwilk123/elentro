"""Data model for the catalogue pipeline.

The pipeline has three stages, and each stage has its own type here:

    RawRow            -- one CSV line, exactly as it was read (never modified)
      |  normalize.py
    NormalizedRecord  -- same row, with derived/parsed fields added
      |  resolve.py
    Product           -- one real-world product, built from 1..n records
      |  harvest.py
    Product.enrichment -- text scraped from the manufacturer's page

Two rules drive the whole design:

1. The raw value is never thrown away. Parsed values (Package, Price,
   Attributes) carry the string they came from, and every product points back
   at its source rows in `Catalogue.records`, so any normalization decision can
   be checked against the original CSV line.
2. Whenever a value was *not* read from the row but guessed by us, we say so in
   a `*_source` field. An inferred category must never look like a stated one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

# The five categories every raw category label collapses into (see normalize.py).
# The label is the human-readable form; it is what gets embedded for search,
# because "LAB_PLASTICWARE" carries much less meaning to a model than the words
# a customer would actually type.
CATEGORY_LABELS = {
    "NUCLEIC_ACID_ISOLATION": "izolacja kwasow nukleinowych, nucleic acid isolation",
    "PCR_REAGENT": "odczynniki do PCR, PCR reagents",
    "LAB_CHEMICAL": "chemia laboratoryjna, odczynniki chemiczne, lab chemicals",
    "LAB_PLASTICWARE": "plastik laboratoryjny, sprzet jednorazowy, lab plasticware",
    "LAB_EQUIPMENT": "aparatura pomiarowa, sprzet pomiarowy, measuring equipment",
}

# Canonical package units, grouped so we can spot nonsense like plasticware
# sold "by the kilogram". Units in different classes are never interchangeable.
UNIT_CLASSES = {
    "reaction": "count",
    "test": "count",
    "item": "count",
    "pack": "count",
    "gram": "mass",
    "kilogram": "mass",
    "milliliter": "volume",
    "liter": "volume",
    "unknown": "unknown",
}


@dataclass(frozen=True)
class Issue:
    """A data-quality problem. We report these; we never silently fix them."""

    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"


# ---------------------------------------------------------------------------
# Stage 1: the untouched input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawRow:
    """One row of katalog_probka.csv, verbatim. This is our audit trail."""

    row_number: int  # 1-based, header excluded
    name: str
    sku: str
    manufacturer: str
    category: str
    package: str
    price: str
    attributes: str


# ---------------------------------------------------------------------------
# Stage 2: parsed field values
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Package:
    """`50 rxn` -> quantity=50, unit='reaction'.

    Quantity and unit stay separate on purpose: "50 reactions" and "50 items"
    share a number but are not the same package, so we never merge them.
    """

    quantity: Decimal | None
    unit: str  # a key of UNIT_CLASSES
    raw: str

    @property
    def unit_class(self) -> str:
        return UNIT_CLASSES[self.unit]

    def __str__(self) -> str:
        qty = "?" if self.quantity is None else f"{self.quantity:g}"
        return f"{qty} {self.unit}"


@dataclass(frozen=True)
class Price:
    """Money is Decimal, never float, and a missing price is None, never 0."""

    amount: Decimal | None
    currency: str | None  # ISO code, only 'PLN' occurs in this dataset
    currency_inferred: bool  # True when the row had a number but no currency
    raw: str

    def __str__(self) -> str:
        if self.amount is None:
            return "brak ceny"
        suffix = " (waluta domyslna)" if self.currency_inferred else ""
        return f"{self.amount} {self.currency}{suffix}"


@dataclass(frozen=True)
class Attributes:
    """Structured view of the free-text `atrybuty_dodatkowe` column."""

    storage_temperature_c: int | None = None
    room_temperature: bool = False
    light_sensitive: bool = False
    dry_ice_required: bool = False
    shelf_life_months: int | None = None
    raw: str = ""

    def is_empty(self) -> bool:
        return not self.raw.strip()

    def understood_nothing(self) -> bool:
        """True when there was text but no rule recognised any part of it."""
        return not self.is_empty() and self == Attributes(raw=self.raw)

    def merged_with(self, other: Attributes) -> Attributes:
        """Combine two descriptions of the same product, flag by flag.

        Whole-object "first non-empty wins" would drop information: of two rows
        saying "chronic przed swiatlem" and "wymaga suchego lodu", only one
        would survive. A flag set on any row is set on the product.
        """
        return Attributes(
            storage_temperature_c=_first(self.storage_temperature_c, other.storage_temperature_c),
            room_temperature=self.room_temperature or other.room_temperature,
            light_sensitive=self.light_sensitive or other.light_sensitive,
            dry_ice_required=self.dry_ice_required or other.dry_ice_required,
            shelf_life_months=_first(self.shelf_life_months, other.shelf_life_months),
            raw=self.raw or other.raw,
        )


def _first(value, fallback):
    """The first value that was actually stated. Conflicts are reported by
    resolve.py, so here the earlier row simply wins."""
    return fallback if value is None else value


# ---------------------------------------------------------------------------
# Stage 2: the normalized row
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizedRecord:
    """One CSV row after deterministic normalization. Still one row per record."""

    raw: RawRow
    sku: str  # uppercased, hyphens/whitespace standardized
    name: str
    manufacturer: str
    category: str | None  # a key of CATEGORY_LABELS, or None if unknown
    category_source: str  # 'row' | 'inferred_from_product_name' | 'missing'
    package: Package
    price: Price
    attributes: Attributes
    issues: tuple[Issue, ...] = ()


# ---------------------------------------------------------------------------
# Stage 3: the resolved product
# ---------------------------------------------------------------------------


@dataclass
class Enrichment:
    """What the manufacturer's page said about a product we already had."""

    source_sku: str  # SKU as written in the HTML (may contain the typo)
    name: str  # manufacturer's name for it -- may differ from the catalogue
    description: str
    storage: str = ""
    applications: str = ""
    match_rule: str = "exact_sku"  # or 'confusable_character_correction'


@dataclass
class Product:
    """One real-world product. Built from one or more NormalizedRecords."""

    canonical_sku: str
    name: str
    manufacturer: str
    category: str | None
    category_source: str
    package: Package
    price: Price
    attributes: Attributes

    # Lineage: which rows produced this product, and why they were merged.
    source_rows: list[int] = field(default_factory=list)
    alias_skus: list[str] = field(default_factory=list)
    resolution_notes: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    enrichment: Enrichment | None = None

    @property
    def all_skus(self) -> list[str]:
        return [self.canonical_sku, *self.alias_skus]

    @property
    def enrichment_status(self) -> str:
        return "enriched" if self.enrichment else "not_found_in_snapshot"


@dataclass
class ExternalCandidate:
    """An HTML card with no counterpart in the catalogue.

    We keep it, but outside the catalogue: the snapshot alone is not enough
    evidence to create a sellable product.
    """

    sku: str
    name: str
    description: str
    storage: str = ""
    applications: str = ""
    status: str = "unmatched_external_candidate"


@dataclass
class Catalogue:
    """Everything the pipeline produced -- the object the CLI works with."""

    products: list[Product]
    external_candidates: list[ExternalCandidate]
    records: list[NormalizedRecord]  # all 228, kept for lineage

    def by_sku(self) -> dict[str, Product]:
        """Lookup table covering canonical SKUs *and* aliases."""
        return {sku: p for p in self.products for sku in p.all_skus}
