"""Tests for the pipeline, mostly end to end against the real sample files.

They pin the decisions a reviewer would question -- how many products came out,
what happened to each duplicate, how the typo in the snapshot was resolved --
rather than the internals of each parser. A few cases run the pipeline over a
small synthetic CSV instead, to reach rules the sample data does not trigger.
Run with `uv run pytest`.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from katalog.models import Attributes
from katalog.normalize import parse_package, parse_price
from katalog.pipeline import build_catalogue
from katalog.store import CatalogueStore


@pytest.fixture(scope="session")
def catalogue():
    return build_catalogue()


@pytest.fixture(scope="session")
def catalogue_store(catalogue, tmp_path_factory):
    store = CatalogueStore(tmp_path_factory.mktemp("store") / "catalogue.sqlite3")
    store.seed(catalogue)
    return store


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, quantity, unit",
    [
        ("50 rxn", 50, "reaction"),
        ("50 reactions", 50, "reaction"),
        ("50szt", 50, "item"),
        ("50 szt.", 50, "item"),
        ("x50", 50, "item"),
        ("50-pack", 50, "item"),
        ("50 test.", 50, "test"),
        ("50 op", 50, "pack"),
        ("50 kg", 50, "kilogram"),
        ("25 ml", 25, "milliliter"),
        # A bare number and a dash are quantities without a unit. We keep the
        # quantity and admit we do not know what it counts.
        ("50", 50, "unknown"),
        ("5 -", 5, "unknown"),
    ],
)
def test_package_variants_keep_their_unit(raw, quantity, unit):
    package = parse_package(raw)
    assert (package.quantity, package.unit) == (Decimal(quantity), unit)
    assert package.raw == raw


def test_price_records_whether_the_currency_was_stated():
    assert parse_price("1021 zl").amount == Decimal("1021")
    assert parse_price("1021 zl").currency_inferred is False
    assert parse_price("1990,00").amount == Decimal("1990.00")
    assert parse_price("1990,00").currency_inferred is True
    # A missing price is missing. It must never become 0.
    assert parse_price("").amount is None


def test_price_refuses_to_guess():
    """A price we cannot read in full is missing, not approximately right."""
    assert parse_price("1 990,00 PLN").amount == Decimal("1990.00")
    assert parse_price("EUR 10").amount is None  # not our currency
    assert parse_price("-10 PLN").amount is None  # a negative price is a data error
    assert parse_price("3399 PLN plus VAT").amount is None  # unread text left over


def test_missing_categories_are_inferred_and_labelled_as_such(catalogue):
    inferred = [
        r
        for r in catalogue.records
        if r.category_source == "inferred_from_product_name"
    ]
    assert len(inferred) == 46  # every empty category in the sample
    assert all(record.category is not None for record in inferred)
    assert all(record.raw.category == "" for record in inferred)


# ---------------------------------------------------------------------------
# Duplicate resolution
# ---------------------------------------------------------------------------


def test_catalogue_size(catalogue):
    assert len(catalogue.records) == 228  # every source row is kept
    assert len(catalogue.products) == 214  # 10 exact + 4 near-duplicate merges


def test_duplicate_rows_combine_information(catalogue):
    """PO-10022 has two rows; only one of them carries the dry-ice attribute."""
    product = catalogue.by_sku()["PO-10022"]
    assert product.source_rows == [1, 128]
    assert product.attributes.dry_ice_required is True


def test_duplicate_rows_combine_attribute_flags():
    """Flags from every row survive; picking one row whole would lose the other."""
    cold = Attributes(storage_temperature_c=-20, raw="przechowywanie -20C")
    dry_ice = Attributes(dry_ice_required=True, raw="wymaga suchego lodu")
    combined = cold.merged_with(dry_ice)
    assert combined.storage_temperature_c == -20
    assert combined.dry_ice_required is True


def test_near_duplicate_becomes_an_alias_not_a_deletion(catalogue):
    by_sku = catalogue.by_sku()
    product = by_sku["CH-10248"]
    assert product.alias_skus == ["CH-10248A"]
    assert by_sku["CH-10248A"] is product  # the alias still resolves
    assert product.source_rows == [36, 222]  # both rows are still traceable


def test_merging_a_variant_keeps_the_issues_found_on_it(catalogue):
    """Both rows are flagged for selling an ELISA reader "by weight" (10 g)."""
    product = catalogue.by_sku()["CH-10248"]
    implausible = [i for i in product.issues if i.code == "package_unit_implausible"]
    assert len(implausible) == 2  # one per source row, not one per surviving product


def test_a_variant_hands_over_the_field_only_it_knows(tmp_path):
    """The base row has no price; the variant does. Merging must not lose it."""
    catalogue = build_catalogue(
        _mini_csv(tmp_path, price_of_variant="99 PLN"), html_path=None
    )
    assert len(catalogue.products) == 1
    assert catalogue.products[0].price.amount == Decimal("99")


def test_units_we_could_not_parse_are_not_treated_as_equal(tmp_path):
    """Two packages nobody understood are not evidence of being the same thing.

    Both parse to "50 unknown". Comparing only the parsed form would merge a
    product sold in bottles with one sold in boxes.
    """
    catalogue = build_catalogue(
        _mini_csv(tmp_path, package_of_variant="50 kartonow"), html_path=None
    )
    assert len(catalogue.products) == 2
    codes = [i.code for p in catalogue.products for i in p.issues]
    assert "variant_sku_not_merged" in codes


def _mini_csv(tmp_path, price_of_variant="", package_of_variant="50 butelek"):
    """A two-row catalogue: XX-10001 and its variant XX-10001A."""
    path = tmp_path / "mini.csv"
    path.write_text(
        "nazwa,nr_katalogowy,producent,kategoria,opakowanie,cena,atrybuty_dodatkowe\n"
        "Bufor X,XX-10001,ACME,Odczynniki,50 butelek,,\n"
        f"Bufor X,XX-10001A,ACME,Odczynniki,{package_of_variant},{price_of_variant},\n",
        encoding="utf-8",
    )
    return path


def test_similar_skus_that_are_different_products_stay_separate(catalogue):
    """The counterexample that rules out edit-distance matching.

    CH-10248 and CH-10258 differ by one character and are not the same product.
    """
    by_sku = catalogue.by_sku()
    assert by_sku["CH-10248"] is not by_sku["CH-10258"]
    assert by_sku["CH-10248"].name != by_sku["CH-10258"].name


# ---------------------------------------------------------------------------
# Harvester
# ---------------------------------------------------------------------------


def test_snapshot_enriches_matching_products(catalogue):
    enriched = [p for p in catalogue.products if p.enrichment]
    assert len(enriched) == 9  # 8 exact SKU hits + 1 corrected typo
    assert all(p.enrichment.description for p in enriched)
    # Cards without a <ul class="specs"> are normal in a scrape, not a failure.
    assert {p.canonical_sku for p in enriched if not p.enrichment.applications} == {
        "NO-10024",
        "NO-10110",
        "NO-10316",
    }


def test_sku_typo_is_corrected_only_because_the_result_is_unique(catalogue):
    """NO-103l6 -> NO-10316: lowercase L read as 1, one catalogue hit."""
    product = catalogue.by_sku()["NO-10316"]
    assert product.enrichment.source_sku == "NO-103l6"
    assert product.enrichment.match_rule == "confusable_character_correction"


def test_the_manufacturer_may_rename_a_product_we_already_have(catalogue):
    """Same SKU, different name -- the catalogue name wins, the other is kept."""
    product = catalogue.by_sku()["NO-10064"]
    assert product.name == "Mix do PCR 2x"
    assert product.enrichment.name == "Mix PCR 2x Universal"


def test_snapshot_products_missing_from_the_csv_are_kept_outside_the_catalogue(
    catalogue,
):
    candidates = {c.sku for c in catalogue.external_candidates}
    assert candidates == {"NO-10500", "NO-99999"}
    assert "NO-10500" not in catalogue.by_sku()


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_exact_catalogue_number_beats_everything(catalogue_store):
    results = catalogue_store.search("ch-10248a")
    assert len(results) == 1
    assert results[0].match_type == "exact_sku"
    assert results[0].score == 1.0
    assert results[0].product.canonical_sku == "CH-10248"
    assert "alias" in results[0].explanation


def test_a_filter_applies_to_the_exact_catalogue_number_too(catalogue_store):
    """CH-10248 is a ChemTech product, so asking for it among NovaGen ones fails."""
    assert (
        catalogue_store.search("CH-10248", manufacturer="ChemTech")[0].match_type
        == "exact_sku"
    )
    assert catalogue_store.search("CH-10248", manufacturer="NovaGen") == []


def test_semantic_search_finds_products_by_meaning(catalogue_store):
    """Loads the embedding model, so this is the slow test in the suite."""
    results = catalogue_store.search("zestaw do oczyszczania RNA", limit=5)
    assert all(r.match_type == "semantic" for r in results)
    assert any("RNA" in r.product.name for r in results)
    assert results == sorted(results, key=lambda r: -r.score)

    # An English query must reach the Polish catalogue (the model is multilingual)
    # and the manufacturer filter must be applied before ranking, not after.
    filtered = catalogue_store.search(
        "analytical balance", manufacturer="NovaGen", limit=3
    )
    assert filtered[0].product.name == "Waga analityczna"
    assert all(r.product.manufacturer == "NovaGen Labs" for r in filtered)
    assert "passed filter" in filtered[0].explanation
