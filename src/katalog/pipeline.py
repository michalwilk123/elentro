"""The whole pipeline in one function, so the order of the stages is obvious."""

from __future__ import annotations

from pathlib import Path

from .harvest import enrich, parse_snapshot
from .models import Catalogue
from .normalize import normalize_rows, read_raw_rows
from .resolve import resolve_products

DATA = Path(__file__).resolve().parents[2] / "dane"
CATALOGUE_CSV = DATA / "katalog_probka.csv"
SNAPSHOT_HTML = DATA / "producent_novagen_snapshot.html"


def build_catalogue(
    csv_path: Path = CATALOGUE_CSV,
    html_path: Path | None = SNAPSHOT_HTML,
) -> Catalogue:
    """CSV (+ optional manufacturer snapshot) -> a searchable catalogue.

    Normalization runs before resolution, and resolution before enrichment.
    That order is the point of the design: you cannot decide whether two rows
    are the same product until their fields are comparable, and you cannot
    attach scraped data until you know what a product is.
    """
    records = normalize_rows(read_raw_rows(csv_path))
    products = resolve_products(records)

    external = []
    if html_path is not None:
        external = enrich(products, parse_snapshot(html_path))

    return Catalogue(products=products, external_candidates=external, records=records)
