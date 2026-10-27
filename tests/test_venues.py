import re

import pandas as pd
import pytest

from cricmind import config
from cricmind.data import venues


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Lord's", "lord s"), ("  M.Chinnaswamy  Stadium ", "m chinnaswamy stadium"),
     ("Mérida Oval", "merida oval"), ("", ""), (None, "")],
)  # fmt: skip
def test_slug(raw, expected):
    assert venues.slug(raw) == expected


def test_default_key_drops_a_trailing_city_only_when_it_is_the_city():
    assert venues.default_key("Eden Gardens, Kolkata", "Kolkata") == "eden gardens|kolkata"
    assert venues.default_key("Eden Gardens", "Kolkata") == "eden gardens|kolkata"
    # last segment is not the city: keep it, it names the ground
    assert venues.default_key("County Ground, New Road", "Worcester") == "county ground new road|worcester"
    assert venues.default_key("Eden Gardens, Kolkata", None) == "eden gardens kolkata|"


def test_default_key_is_non_null_for_missing_parts():
    assert venues.default_key(None, None) == "|"
    assert venues.default_key("Ground", None) == "ground|"


def test_missing_venue_name_gets_a_per_match_key_never_a_shared_one():
    assert venues.venue_key(None, None, "123") == "unknown 123|"
    assert venues.venue_key("", "Pune", "124") == "unknown 124|pune"
    assert venues.venue_key(None, "Pune", "125") != venues.venue_key(None, "Pune", "126")
    assert venues.venue_key("Ground", None, "127") == "ground|"  # a name without a city is still a ground


def test_same_name_in_different_cities_is_not_merged():
    keys = {venues.venue_key("County Ground", c, "m") for c in ("Bristol", "Derby", "Hove", "Northampton")}
    assert len(keys) == 4
    karachi, hamilton = (venues.venue_key("National Stadium", c, "m") for c in ("Karachi", "Hamilton"))
    assert karachi != hamilton


@pytest.mark.parametrize(
    "group",
    [
        [("Wankhede Stadium", None), ("Wankhede Stadium", "Mumbai"), ("Wankhede Stadium, Mumbai", "Mumbai")],
        [("M Chinnaswamy Stadium", "Bangalore"), ("M.Chinnaswamy Stadium", "Bengaluru"),
         ("M Chinnaswamy Stadium, Bangalore", "Bengaluru")],
        [("Sheikh Zayed Stadium", "Abu Dhabi"), ("Zayed Cricket Stadium, Abu Dhabi", "Abu Dhabi")],
        [("Zahur Ahmed Chowdhury Stadium", "Chittagong"),
         ("Zahur Ahmed Chowdhury Stadium, Chattogram", "Chattogram")],
        [("County Ground, Hove", "Brighton"), ("County Ground", "Hove")],
    ],
)  # fmt: skip
def test_spellings_of_one_ground_share_a_key(group):
    assert len({venues.venue_key(v, c, "m") for v, c in group}) == 1


def test_alias_wins_over_the_computed_default():
    assert venues.default_key("Wankhede Stadium", None) == "wankhede stadium|"
    assert venues.venue_key("Wankhede Stadium", None, "m") == "wankhede stadium|mumbai"


def test_unaliased_pair_falls_back_to_default():
    assert venues.venue_key("Some New Ground", "Some City", "m") == "some new ground|some city"


def test_alias_file_is_well_formed():
    aliases = venues.load_aliases()  # raises on a duplicate (raw_venue, raw_city)
    assert aliases
    for pair, key in aliases.items():
        assert re.fullmatch(r"[a-z0-9 ]+\|[a-z0-9 ]*", key), f"{pair}: {key!r} is not a normalised key"
        assert key == key.strip() and "  " not in key, f"{pair}: {key!r}"


def test_duplicate_alias_rows_raise(tmp_path):
    f = tmp_path / "aliases.csv"
    f.write_text("raw_venue,raw_city,venue_key,note\nA,,a|x,\nA,,a|y,\n")
    with pytest.raises(ValueError, match="duplicate"):
        venues.load_aliases(f)


def test_every_alias_row_matches_a_pair_in_the_data():
    # A typo in raw_venue/raw_city silently turns an alias into a no-op, so check against the real data.
    path = config.DATA_PROCESSED / "matches.parquet"
    if not path.exists():
        pytest.skip("matches.parquet not built")
    seen = set(pd.read_parquet(path, columns=["venue", "city"]).fillna("").itertuples(index=False, name=None))
    stale = [pair for pair in venues.load_aliases() if pair not in seen]
    assert not stale, f"{len(stale)} alias rows match no match, e.g. {stale[:3]}"
