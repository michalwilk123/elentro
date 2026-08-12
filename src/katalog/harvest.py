"""Stage 3: the mini-harvester for the NovaGen Labs page snapshot.

The snapshot is scraped product data, so it is treated as a *claim* rather than
as truth: it can add a description and specs to a product we already have, but
it never overwrites catalogue data and it cannot create a catalogue product on
its own.

Matching is by catalogue number only, never by name -- one card in this snapshot
deliberately carries a different product name for a SKU we already know, so name
matching would be actively harmful here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product as combinations
from pathlib import Path

from bs4 import BeautifulSoup

from .models import Enrichment, ExternalCandidate, Issue, Product
from .normalize import fold, normalize_sku, normalize_text

# Characters a human (or an OCR pass) mixes up when copying a catalogue number.
# Kept intentionally tiny: every extra pair multiplies the number of catalogue
# numbers we are willing to invent.
CONFUSABLE_CHARACTERS = {"L": "1", "I": "1", "O": "0"}


@dataclass
class HarvestedCard:
    """One <div class="product-card"> from the snapshot, as written there."""

    sku: str
    name: str
    description: str
    storage: str = ""
    applications: str = ""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_snapshot(html_path: Path) -> list[HarvestedCard]:
    """Read the manufacturer page. Missing fields are normal, not an error."""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    return [_parse_card(card) for card in soup.select("div.product-card")]


def _parse_card(card) -> HarvestedCard:
    specs = [normalize_text(item.get_text()) for item in card.select("ul.specs li")]
    return HarvestedCard(
        sku=card.get("data-sku", ""),
        name=_text(card, "h2.product-name"),
        description=_text(card, "p.product-desc"),
        storage=_spec(specs, "Przechowywanie:"),
        applications=_spec(specs, "Zastosowania:"),
    )


def _text(card, selector: str) -> str:
    element = card.select_one(selector)
    return normalize_text(element.get_text()) if element else ""


def _spec(specs: list[str], label: str) -> str:
    """`Przechowywanie: 4°C` -> `4°C`, or empty when the card omits the line."""
    return next((s[len(label) :].strip() for s in specs if s.startswith(label)), "")


# ---------------------------------------------------------------------------
# Merging into the catalogue
# ---------------------------------------------------------------------------


def enrich(products: list[Product], cards: list[HarvestedCard]) -> list[ExternalCandidate]:
    """Attach each card to its catalogue product; return the leftovers.

    Leftovers are kept as external candidates rather than dropped or promoted
    to products: "this SKU is not in the catalogue" is a fact worth passing on
    to whoever maintains the catalogue, but a scraped page is not authority
    enough to add a sellable item.
    """
    by_sku = {sku: p for p in products for sku in p.all_skus}
    unmatched: list[ExternalCandidate] = []

    for card in cards:
        match = match_card_to_sku(card.sku, set(by_sku))
        if match is None:
            unmatched.append(
                ExternalCandidate(
                    sku=normalize_sku(card.sku),
                    name=card.name,
                    description=card.description,
                    storage=card.storage,
                    applications=card.applications,
                )
            )
            continue

        sku, rule = match
        matched = by_sku[sku]
        if rule != "exact_sku":
            # A corrected catalogue number is the one risky decision here, so we
            # write down the corroborating evidence next to the product. The
            # names agreeing is supporting evidence only -- never the match
            # itself, since another card in this snapshot renames a known SKU.
            agrees = "and" if fold(matched.name) == fold(card.name) else "but"
            matched.resolution_notes.append(
                f"snapshot SKU {card.sku!r} read as {sku} (confusable characters, "
                f"unique catalogue hit) {agrees} the product name {card.name!r} "
                f"{'matches' if agrees == 'and' else 'differs from'} {matched.name!r}"
            )

        if matched.enrichment is not None:
            # Two cards claiming the same product. Letting document order pick
            # the winner would hide the ambiguity, so we keep the first and say so.
            matched.issues.append(
                Issue(
                    "snapshot_card_conflict",
                    f"card {card.sku!r} also resolves to {sku}, already enriched from "
                    f"{matched.enrichment.source_sku!r}; kept the first",
                )
            )
            continue

        matched.enrichment = Enrichment(
            source_sku=card.sku,
            name=card.name,
            description=card.description,
            storage=card.storage,
            applications=card.applications,
            match_rule=rule,
        )
        if fold(matched.name) != fold(card.name):
            matched.resolution_notes.append(
                f"manufacturer calls {sku} {card.name!r}; catalogue name kept as "
                f"{matched.name!r}"
            )

    return unmatched


def match_card_to_sku(card_sku: str, known_skus: set[str]) -> tuple[str, str] | None:
    """Find the catalogue SKU a scraped catalogue number refers to.

    Two rules, in order of confidence:

    1. The normalized SKU is in the catalogue -- done.
    2. The SKU is not, but correcting confusable characters produces exactly one
       catalogue number that is. This is how `NO-103l6` resolves to `NO-10316`:
       lowercase L is read as 1. We require the result to be *unique*, because
       an ambiguous correction is no better than a guess.

    Returns (catalogue_sku, rule_name), or None when nothing matches.
    """
    sku = normalize_sku(card_sku)
    if sku in known_skus:
        return sku, "exact_sku"

    corrected = {c for c in _confusable_variants(sku) if c in known_skus}
    if len(corrected) == 1:
        return corrected.pop(), "confusable_character_correction"
    return None


def _confusable_variants(sku: str) -> set[str]:
    """Every SKU you get by fixing confusable characters after the prefix.

    The manufacturer prefix (`NO-`) is left alone: it is short, meaningful and
    not where typos of this kind happen, and freezing it keeps the candidate
    set small enough to reason about.
    """
    prefix, separator, digits = sku.partition("-")
    if not separator:
        return set()

    options = [{c, CONFUSABLE_CHARACTERS.get(c, c)} for c in digits]
    return {f"{prefix}-{''.join(variant)}" for variant in combinations(*options)} - {sku}
