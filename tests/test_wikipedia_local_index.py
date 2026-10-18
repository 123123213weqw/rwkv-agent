from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import zlib

from benchmarks.build_wikipedia_title_index import create_schema
from rwkv_search.realtime.discovery import search_local_wikipedia


def test_local_wikipedia_title_search_returns_grounded_article_text() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        database = Path(temporary) / "wikipedia.sqlite3"
        connection = sqlite3.connect(database)
        create_schema(connection)
        pages = [
            (
                1,
                "Punxsutawney Phil",
                "https://en.wikipedia.org/wiki/Punxsutawney_Phil",
                zlib.compress(
                    b"Punxsutawney Phil is a groundhog in Punxsutawney, Pennsylvania.\n"
                    b"The tradition is associated with Groundhog Day."
                ),
            ),
            (
                2,
                "United States Capitol",
                "https://en.wikipedia.org/wiki/United_States_Capitol",
                zlib.compress(b"The United States Capitol is in Washington, D.C."),
            ),
        ]
        connection.executemany(
            "INSERT INTO pages(rowid,title,url,content_zlib) VALUES(?,?,?,?)",
            pages,
        )
        connection.executemany(
            "INSERT INTO titles(rowid,title) VALUES(?,?)",
            [(row[0], row[1]) for row in pages],
        )
        connection.commit()
        connection.close()

        results = search_local_wikipedia(
            database,
            "How is Punxsutawney Phil related to Groundhog Day?",
        )
        assert results
        assert results[0].url.endswith("/Punxsutawney_Phil")
        assert "groundhog" in results[0].snippet.casefold()
        assert results[0].engine == "wikipedia_local"
