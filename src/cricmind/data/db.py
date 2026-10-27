"""DuckDB views over the Phase 1 parquet files: deliveries, legal_deliveries, matches, players.

from cricmind.data.db import connect
con = connect()
"""

from pathlib import Path

import duckdb

from cricmind import config


def connect(processed_dir: Path | None = None, people_csv: Path | None = None) -> duckdb.DuckDBPyConnection:
    """In-memory connection with views over the parquet/CSV files (nothing is copied or written)."""
    # lean: no persisted .duckdb or read-only mode - Phase 6 (SQL guard) and Phase 8 (demo bundle) add them.
    processed_dir = processed_dir or config.DATA_PROCESSED
    people_csv = people_csv or config.DATA_RAW / config.DATA_SNAPSHOT / "people.csv"
    con = duckdb.connect()
    con.execute(
        f"create view deliveries as select * from read_parquet('{processed_dir / 'deliveries.parquet'}')"
    )
    # Rows every later phase models on (D-004): legal balls outside super overs, so `outcome` is never null.
    con.execute(
        "create view legal_deliveries as select * from deliveries where is_legal and not is_super_over"
    )
    con.execute(f"create view matches as select * from read_parquet('{processed_dir / 'matches.parquet'}')")
    # Registry id -> name, for display only; players and venues are always joined on ids (D-003).
    con.execute(
        f"create view players as select identifier as player_id, name, unique_name "
        f"from read_csv('{people_csv}', header = true)"
    )
    return con
