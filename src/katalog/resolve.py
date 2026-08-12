"""Stage 2: turn 228 normalized rows into one Product per real-world product.

Two different problems, handled separately and in this order:

1. The same SKU appears twice  -> certainly the same product. Merge the rows,
   combining information and recording any field where they disagree.
2. Two SKUs look related        -> maybe the same product. Merge only if every
   other field is identical, and keep both SKUs.

What we deliberately do *not* do is merge by edit distance. In this catalogue
CH-10248 and CH-10258 are two genuinely different products one character apart,
so "distance <= 1 means duplicate" would corrupt the catalogue. The suffix
pattern below is only used to *propose* pairs; the merge itself is decided by
comparing the actual product data.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace

from .models import Attributes, Issue, NormalizedRecord, Package, Price, Product
from .normalize import check_package_plausibility, fold

# A SKU that is a base SKU plus a variant marker: CH-10248A, BI-10220-1.
VARIANT_SKU = re.compile(r"^(?P<base>[A-Z]{2}-\d+)(?P<suffix>[A-Z]|-\d+)$")


def resolve_products(records: list[NormalizedRecord]) -> list[Product]:
    """Records -> products. Nothing is deleted; duplicates become aliases."""
    products = [_merge_rows_with_same_sku(group) for group in _group_by_sku(records)]
    return _merge_variant_skus(products)


# ---------------------------------------------------------------------------
# 1. Exact duplicates: the same SKU on more than one row
# ---------------------------------------------------------------------------


def _group_by_sku(records: list[NormalizedRecord]) -> list[list[NormalizedRecord]]:
    groups: dict[str, list[NormalizedRecord]] = defaultdict(list)
    for record in records:
        groups[record.sku].append(record)
    return list(groups.values())


def _merge_rows_with_same_sku(group: list[NormalizedRecord]) -> Product:
    """Combine rows sharing a SKU: first non-empty value wins, conflicts logged.

    "First" means lowest CSV row number, so the result does not depend on
    dictionary ordering. In this sample only PO-10022 actually gains anything:
    one of its two rows has the dry-ice attribute and the other is blank.
    """
    group = sorted(group, key=lambda record: record.raw.row_number)
    first = group[0]

    # Attributes are the one field where "first non-empty wins" would lose
    # information, so the flags of every row are combined.
    attributes = first.attributes
    for record in group[1:]:
        attributes = attributes.merged_with(record.attributes)
    # Whichever row supplied the category also supplies its provenance label.
    categorised = next((record for record in group if record.category), first)

    product = Product(
        canonical_sku=first.sku,
        name=_pick(group, "name"),
        manufacturer=_pick(group, "manufacturer"),
        category=categorised.category,
        category_source=categorised.category_source,
        package=_pick(group, "package"),
        price=_pick(group, "price"),
        attributes=attributes,
        source_rows=[record.raw.row_number for record in group],
        issues=[issue for record in group for issue in record.issues],
    )

    if len(group) > 1:
        product.resolution_notes.append(
            f"exact duplicate SKU: merged rows {product.source_rows} "
            f"(first non-empty value per field)"
        )
        product.issues.extend(_conflicts(group))

    # Checked on the values that ended up on the product, which are not
    # necessarily the values of the first row.
    product.issues.extend(check_package_plausibility(product.category, product.package))
    return product


def _pick(group: list[NormalizedRecord], field: str):
    """First non-empty value of `field` across the group."""
    values = [getattr(record, field) for record in group]
    return next((value for value in values if not _is_empty(value)), values[0])


def _conflicts(group: list[NormalizedRecord]) -> list[Issue]:
    """Report fields where two rows state different, non-empty values."""
    issues = []
    for field in ("name", "manufacturer", "category", "package", "price", "attributes"):
        stated = {_comparable(getattr(record, field)) for record in group}
        stated.discard(None)
        if len(stated) > 1:
            issues.append(
                Issue("duplicate_field_conflict", f"{field} differs between rows: {sorted(stated)}")
            )
    return issues


# ---------------------------------------------------------------------------
# 2. Near-duplicates: BASE vs BASE+suffix
# ---------------------------------------------------------------------------


def _merge_variant_skus(products: list[Product]) -> list[Product]:
    """Fold `CH-10248A` into `CH-10248` when the two describe the same thing."""
    by_sku = {product.canonical_sku: product for product in products}
    merged_away: set[str] = set()

    for product in products:
        match = VARIANT_SKU.match(product.canonical_sku)
        base = by_sku.get(match.group("base")) if match else None
        # A base that is itself an alias of something else would build a chain
        # (A -> B -> C). That does not occur here, and resolving it silently is
        # exactly the kind of hidden decision this module avoids.
        if base is None or base is product or base.canonical_sku in merged_away:
            continue

        differences = _differences(base, product)
        if differences:
            # Related-looking SKUs that are not the same product: leave both,
            # but say why we did not merge them.
            product.issues.append(
                Issue(
                    "variant_sku_not_merged",
                    f"{product.canonical_sku} looks like a variant of "
                    f"{base.canonical_sku} but {', '.join(differences)} differ",
                )
            )
            continue

        _absorb(base, product)
        merged_away.add(product.canonical_sku)

    return [p for p in products if p.canonical_sku not in merged_away]


def _absorb(base: Product, variant: Product) -> None:
    """Move everything the variant carried onto the base, then let it go.

    A field the base is missing is taken from the variant -- the two passed the
    conjunctive rule, so where one is silent the other is the only evidence we
    have. Issues and notes come along too, otherwise a data-quality problem
    would disappear just because the row it was found on was merged.
    """
    for name in ("name", "manufacturer", "package", "price", "attributes"):
        if _is_empty(getattr(base, name)):
            setattr(base, name, getattr(variant, name))

    if base.category is None and variant.category is not None:
        base.category, base.category_source = variant.category, variant.category_source

    base.alias_skus.extend([variant.canonical_sku, *variant.alias_skus])
    base.source_rows = sorted(base.source_rows + variant.source_rows)
    base.issues.extend(variant.issues)
    base.resolution_notes.extend(variant.resolution_notes)
    base.resolution_notes.append(
        f"near-duplicate: {variant.canonical_sku} kept as an alias of "
        f"{base.canonical_sku}; all normalized fields are identical"
    )


def _differences(base: Product, variant: Product) -> list[str]:
    """Which fields stop these two from being the same product?

    Every field must agree -- this is a conjunctive rule, not a score. An empty
    value on one side counts as agreement (it adds no evidence against).
    """
    differing = []
    for field in ("name", "manufacturer", "category", "package", "price", "attributes"):
        left = _comparable(getattr(base, field))
        right = _comparable(getattr(variant, field))
        if left is not None and right is not None and left != right:
            differing.append(field)
    return differing


# ---------------------------------------------------------------------------
# Comparing field values
# ---------------------------------------------------------------------------


def _is_empty(value) -> bool:
    return _comparable(value) is None


def _comparable(value) -> str | None:
    """A field's value as a comparable string, or None when it says nothing."""
    if value is None:
        return None
    if isinstance(value, Price):
        return None if value.amount is None else f"{value.amount} {value.currency}"
    if isinstance(value, str):
        return value.casefold() or None

    # For the two parsed types, anything a parser did not understand falls back
    # to the raw text. Otherwise "50 bottles" and "50 boxes" would both read as
    # "50 unknown" and merge, which is the opposite of being careful.
    if isinstance(value, Package):
        if value.quantity is None and value.unit == "unknown":
            return None
        return str(value) if value.unit != "unknown" else f"{value} of {fold(value.raw)}"
    if isinstance(value, Attributes):
        if value.is_empty():
            return None
        return fold(value.raw) if value.understood_nothing() else str(replace(value, raw=""))
    raise TypeError(f"no comparison rule for {type(value).__name__}")
