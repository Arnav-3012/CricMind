"""db.connect: the four views exist and legal_deliveries filters as D-004 says. Synthetic files only."""

import pandas as pd
import pytest

from cricmind.data.db import connect


@pytest.fixture()
def con(tmp_path):
    deliveries = pd.DataFrame(
        {
            "match_id": ["m1"] * 4,
            "is_legal": [True, False, True, True],
            "is_super_over": [False, False, False, True],
            "outcome": ["dot", None, "4", "6"],
        }
    )
    deliveries.to_parquet(tmp_path / "deliveries.parquet")
    pd.DataFrame({"match_id": ["m1"]}).to_parquet(tmp_path / "matches.parquet")
    (tmp_path / "people.csv").write_text(
        "identifier,name,unique_name,key_cricinfo\nabc123,A Player,A Player,1\n"
    )
    return connect(tmp_path, tmp_path / "people.csv")


def test_views_exist(con):
    names = {r[0] for r in con.execute("select view_name from duckdb_views() where not internal").fetchall()}
    assert names == {"deliveries", "legal_deliveries", "matches", "players"}


def test_legal_deliveries_excludes_extras_and_super_overs(con):
    assert con.execute("select count(*) from deliveries").fetchone()[0] == 4
    assert con.execute("select outcome from legal_deliveries order by outcome").fetchall() == [
        ("4",),
        ("dot",),
    ]


def test_players_exposes_registry_id_and_name(con):
    assert con.execute("select player_id, name from players").fetchall() == [("abc123", "A Player")]
