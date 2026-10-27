"""Venue join/model key (D-027): normalised (venue, city) slug + a hand-reviewed alias file.

Cricsheet venue names are free text (spellings, missing or renamed cities) and only ~1% of matches carry a
registry id, so `venue_key` is the venue key everywhere downstream; `venue_id` stays the raw registry id.
Alias rule: if unsure, do not merge. An unmerged ground only splits samples; a wrong merge mixes two grounds.
"""

import csv
import re
import unicodedata
from functools import cache
from pathlib import Path

ALIASES_CSV = Path(__file__).with_name("venue_aliases.csv")
SEP = "|"


def slug(text: str | None) -> str:
    """NFKD -> ASCII, lowercase, punctuation -> space, collapsed. None -> ''."""
    ascii_text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def _drop_trailing_city(venue: str, city: str | None) -> str:
    """'Eden Gardens, Kolkata' + city Kolkata -> 'Eden Gardens'; only when the last segment IS the city."""
    head, comma, tail = venue.rpartition(",")
    if comma and slug(city) and slug(tail) == slug(city) and slug(head):
        return head
    return venue


def default_key(venue: str | None, city: str | None) -> str:
    return slug(_drop_trailing_city(venue or "", city)) + SEP + slug(city)


@cache
def load_aliases(path: Path = ALIASES_CSV) -> dict[tuple[str, str], str]:
    """(raw_venue, raw_city) -> venue_key. Raw strings are matched exactly; a null city is ''."""
    aliases: dict[tuple[str, str], str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pair = (row["raw_venue"], row["raw_city"])
            if pair in aliases:
                raise ValueError(f"duplicate alias row for {pair}")
            aliases[pair] = row["venue_key"]
    return aliases


def venue_key(venue: str | None, city: str | None, match_id: str) -> str:
    """Non-null key. The alias file wins over the computed default. No venue name -> a per-match key, so two
    unknown grounds are never merged (a city alone does not name a ground); D-028."""
    key = load_aliases().get((venue or "", city or "")) or default_key(venue, city)
    return f"unknown {slug(match_id)}{key}" if key.startswith(SEP) else key
