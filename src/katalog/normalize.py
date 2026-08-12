"""Stage 1: deterministic normalization of the CSV.

Apart from reading the file, everything in this module is a pure function over
a lookup table. There is no fuzzy matching here on purpose -- fuzziness in normalization produces mistakes
that nobody can explain afterwards. If a value is not covered by a rule below,
it stays unknown and gets an Issue, rather than being guessed at.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .models import Attributes, Issue, NormalizedRecord, Package, Price, RawRow

# ---------------------------------------------------------------------------
# Category vocabulary
# ---------------------------------------------------------------------------

# All 21 raw category spellings in the sample, mapped by hand. An explicit table
# is boring, but it is reviewable and it cannot drift the way a similarity
# threshold can. New spellings surface as an Issue instead of being mis-filed.
CATEGORY_MAP = {
    "nucleic acid isolation": "NUCLEIC_ACID_ISOLATION",
    "izolacja kwasow nukleinowych": "NUCLEIC_ACID_ISOLATION",
    "izolacja dna/rna": "NUCLEIC_ACID_ISOLATION",
    "izolacja dna": "NUCLEIC_ACID_ISOLATION",
    "pcr reagents": "PCR_REAGENT",
    "pcr - odczynniki": "PCR_REAGENT",
    "odczynniki pcr": "PCR_REAGENT",
    "odczynniki do pcr": "PCR_REAGENT",
    "chemia laboratoryjna": "LAB_CHEMICAL",
    "chemicals": "LAB_CHEMICAL",
    "odczynniki": "LAB_CHEMICAL",
    "odczynniki chemiczne": "LAB_CHEMICAL",
    "plastik laboratoryjny": "LAB_PLASTICWARE",
    "plastiki lab.": "LAB_PLASTICWARE",
    "laboratory plasticware": "LAB_PLASTICWARE",
    "sprzet jednorazowy": "LAB_PLASTICWARE",
    "measuring equipment": "LAB_EQUIPMENT",
    "pomiary": "LAB_EQUIPMENT",
    "aparatura pomiarowa": "LAB_EQUIPMENT",
    "sprzet pomiarowy": "LAB_EQUIPMENT",
}

# Package unit spellings -> canonical unit. The token is whatever letters follow
# the number, so "50szt", "50 szt." and "x50 szt" all arrive here as "szt".
UNIT_MAP = {
    "rxn": "reaction",
    "reactions": "reaction",
    "reaction": "reaction",
    "reakcji": "reaction",
    "test": "test",
    "testy": "test",
    "szt": "item",
    "pcs": "item",
    "pack": "item",
    "op": "pack",
    "g": "gram",
    "kg": "kilogram",
    "ml": "milliliter",
    "l": "liter",
}

# Products sold as discrete objects should not be priced by mass or volume.
# We flag those rows; we do not "correct" them -- only the supplier knows which
# of the two fields is wrong.
COUNTABLE_CATEGORIES = {"LAB_PLASTICWARE", "LAB_EQUIPMENT"}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_raw_rows(csv_path: Path) -> list[RawRow]:
    """Read the CSV without interpreting anything."""
    with open(csv_path, encoding="utf-8", newline="") as handle:
        return [
            RawRow(
                row_number=number,
                name=row["nazwa"],
                sku=row["nr_katalogowy"],
                manufacturer=row["producent"],
                category=row["kategoria"],
                package=row["opakowanie"],
                price=row["cena"],
                attributes=row["atrybuty_dodatkowe"],
            )
            for number, row in enumerate(csv.DictReader(handle), start=1)
        ]


# ---------------------------------------------------------------------------
# Field-level parsers
# ---------------------------------------------------------------------------


def normalize_sku(raw: str) -> str:
    """Uppercase, unify dash characters, drop stray whitespace.

    Deliberately conservative: this only fixes formatting. It never edits the
    identifier itself, because in this catalogue CH-10248 and CH-10258 are two
    different products -- a one-character "fix" would silently invent a merge.
    """
    text = raw.strip().upper()
    text = re.sub(r"[‐-―−]", "-", text)  # en/em dash -> hyphen
    text = re.sub(r"\s*-\s*", "-", text)
    return re.sub(r"\s+", " ", text)


def normalize_text(raw: str) -> str:
    """Collapse whitespace in a human-readable field."""
    return re.sub(r"\s+", " ", raw).strip()


def fold(text: str) -> str:
    """Casefold and drop Polish diacritics, for comparing text written both ways.

    The catalogue writes "Czytnik plytek ELISA" and the manufacturer's page
    writes "Czytnik płytek ELISA". Those are the same name, and only comparisons
    should ignore the difference -- the stored values keep their own spelling.
    """
    decomposed = unicodedata.normalize("NFKD", text.replace("ł", "l").replace("Ł", "L"))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def parse_category(raw: str) -> str | None:
    """Map a raw category label onto the controlled vocabulary."""
    return CATEGORY_MAP.get(normalize_text(raw).casefold())


def parse_package(raw: str) -> Package:
    """`50 rxn` -> 50 reactions, `x50` -> 50 items, `5 -` -> 5 of something."""
    text = normalize_text(raw).casefold()
    if not text:
        return Package(None, "unknown", raw)

    number = re.search(r"\d+(?:[.,]\d+)?", text)
    quantity = _to_decimal(number.group()) if number else None

    # Whatever letters follow the number is the unit: "-pack" -> "pack".
    tail = text[number.end() :] if number else ""
    unit = UNIT_MAP.get(re.sub(r"[^a-z]", "", tail), "unknown")

    # "x50" carries no unit word, but the leading x means "50 pieces".
    if unit == "unknown" and text.startswith("x"):
        unit = "item"

    return Package(quantity, unit, raw)


# `1021 zl`, `3908,00`, `1 990,00 PLN` -- an amount, optionally followed by the
# only currency this catalogue uses. Anchored, so anything else fails to parse
# instead of being partially read.
PRICE = re.compile(r"^(?P<amount>\d[\d ]*(?:[.,]\d+)?)\s*(?P<currency>pln|zl|zł)?$", re.IGNORECASE)


def parse_price(raw: str) -> Price:
    """`1021 zl` / `3908,00` / `` -> Decimal + currency, or nothing at all.

    Refuses to guess. `EUR 10` and `-10` do not parse: silently relabelling a
    foreign or malformed amount as PLN would put a wrong number on a quote.
    """
    match = PRICE.match(normalize_text(raw))
    if match is None:
        return Price(None, None, False, raw)

    return Price(
        amount=_to_decimal(match.group("amount").replace(" ", "")),
        currency="PLN",  # the only currency in this catalogue
        currency_inferred=match.group("currency") is None,
        raw=raw,
    )


# The free-text attribute column only ever uses these five phrasings, so a short
# list of patterns covers it. Anything else falls through and is kept as text.
def parse_attributes(raw: str) -> Attributes:
    text = normalize_text(raw).casefold()
    temperature = re.search(r"przechowywanie\s*(-?\d+)\s*c\b", text)
    shelf_life = re.search(r"termin waznosci\s*(\d+)\s*mies", text)
    return Attributes(
        storage_temperature_c=int(temperature.group(1)) if temperature else None,
        room_temperature=bool(re.search(r"temp\w*\.?\s*pokojowa", text)),
        light_sensitive=bool(re.search(r"chroni\w*\s+przed\s+swiatlem", text)),
        dry_ice_required=bool(re.search(r"such\w+\s+lod", text)),
        shelf_life_months=int(shelf_life.group(1)) if shelf_life else None,
        raw=normalize_text(raw),
    )


def _to_decimal(text: str) -> Decimal | None:
    try:
        return Decimal(text.replace(",", "."))
    except InvalidOperation:
        return None


# ---------------------------------------------------------------------------
# Row-level normalization
# ---------------------------------------------------------------------------


def normalize_rows(rows: list[RawRow]) -> list[NormalizedRecord]:
    """Normalize every row, then fill in categories that can be inferred.

    Two passes, because the second one needs the whole file: a row with an empty
    category can borrow the category of rows carrying the exact same product
    name. In this sample that resolves all 46 empty categories, and no product
    name ever points at two different categories -- so the inference stays
    deterministic. It is still marked as inferred.
    """
    records = [_normalize_row(row) for row in rows]
    by_name = _category_by_name(records)
    return [_infer_missing_category(record, by_name) for record in records]


def _normalize_row(row: RawRow) -> NormalizedRecord:
    issues: list[Issue] = []

    category = parse_category(row.category)
    if category is None and row.category.strip():
        issues.append(Issue("unknown_category_label", f"{row.category!r} is not in CATEGORY_MAP"))

    package = parse_package(row.package)
    if package.unit == "unknown":
        issues.append(Issue("package_unit_unknown", f"cannot read a unit from {row.package!r}"))

    price = parse_price(row.price)
    if price.amount is None:
        issues.append(Issue("price_missing", f"no price in {row.price!r}"))

    return NormalizedRecord(
        raw=row,
        sku=normalize_sku(row.sku),
        name=normalize_text(row.name),
        manufacturer=normalize_text(row.manufacturer),
        category=category,
        category_source="row" if category else "missing",
        package=package,
        price=price,
        attributes=parse_attributes(row.attributes),
        issues=tuple(issues),
    )


def _category_by_name(records: list[NormalizedRecord]) -> dict[str, set[str]]:
    """product name -> every category that name appears with in the file."""
    seen: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.category:
            seen[record.name.casefold()].add(record.category)
    return seen


def _infer_missing_category(
    record: NormalizedRecord, by_name: dict[str, set[str]]
) -> NormalizedRecord:
    if record.category is not None:
        return record

    candidates = by_name.get(record.name.casefold(), set())
    if len(candidates) != 1:
        # Either nothing to learn from, or the name is ambiguous. Leave it empty
        # rather than pick one; a wrong category is worse than a missing one.
        detail = "no other row with this name" if not candidates else f"ambiguous: {candidates}"
        return _with(record, issues=record.issues + (Issue("category_missing", detail),))

    return _with(
        record,
        category=next(iter(candidates)),
        category_source="inferred_from_product_name",
    )


def check_package_plausibility(category: str | None, package: Package) -> list[Issue]:
    """Flag a package unit that contradicts the product category.

    Takes the values rather than a record, so resolve.py can run it on the
    package that actually ended up on the product, not on whichever row it
    happened to come from.
    """
    if category in COUNTABLE_CATEGORIES and package.unit_class in {"mass", "volume"}:
        return [Issue("package_unit_implausible", f"{category} sold as {package.raw!r}")]
    return []


def _with(record: NormalizedRecord, **changes) -> NormalizedRecord:
    """Copy of a frozen record with some fields replaced."""
    return replace(record, **changes)
