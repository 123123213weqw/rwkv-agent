from __future__ import annotations

import bz2
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
import zlib


SCRIPT = Path(__file__).parents[1] / "benchmarks" / "build_wikipedia_xml_index.py"
SPEC = importlib.util.spec_from_file_location("wikipedia_xml_index", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def xml_page(page_id: int, title: str, text: str, *, redirect: bool = False) -> str:
    redirect_value = '<redirect title="Target" />' if redirect else ""
    return (
        f"<page><title>{title}</title><ns>0</ns><id>{page_id}</id>{redirect_value}"
        f"<revision><id>{page_id + 100}</id><text>{text}</text></revision></page>"
    )


class WikipediaXMLIndexTests(unittest.TestCase):
    def test_build_and_resume_produce_search_compatible_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "enwiki-20260701-pages-articles1.xml-test.bz2"
            payload = (
                '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">'
                + xml_page(
                    1,
                    "Example Person",
                    "'''Example Person''' won the [[Sample Award|award]] in 2024. {{cite}}",
                )
                + xml_page(2, "Redirect", "ignored", redirect=True)
                + "</mediawiki>"
            ).encode()
            shard.write_bytes(bz2.compress(payload))
            status = root / "dumpstatus.json"
            metadata = {
                shard.name: {
                    "size": shard.stat().st_size,
                    "sha1": hashlib.sha1(shard.read_bytes()).hexdigest(),
                    "url": "/example",
                }
            }
            status.write_text(
                json.dumps({"jobs": {"articlesdump": {"status": "done", "files": metadata}}})
            )
            database = root / "index.sqlite3"
            manifest = root / "manifest.json"
            first = MODULE.build(root, status, database, manifest, max_article_chars=50_000)
            self.assertEqual(first["rows"], 1)
            connection = sqlite3.connect(database)
            row = connection.execute(
                "SELECT pages.title,pages.content_zlib FROM titles JOIN pages ON pages.rowid=titles.rowid WHERE titles MATCH 'Example'"
            ).fetchone()
            connection.close()
            self.assertEqual(row[0], "Example Person")
            self.assertIn("won the award", zlib.decompress(row[1]).decode())

            # A missing final manifest is the only resumable state; completed
            # source transactions are skipped rather than duplicated.
            manifest.unlink()
            second = MODULE.build(root, status, database, manifest, max_article_chars=50_000)
            self.assertEqual(second["rows"], 1)
            self.assertEqual(len(second["inputs"]), 1)

    def test_partial_ingest_does_not_write_final_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "one.bz2"
            shard.write_bytes(
                bz2.compress(
                    (
                        '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">'
                        + xml_page(7, "Partial Page", "Partial page body.")
                        + "</mediawiki>"
                    ).encode()
                )
            )
            missing = root / "two.bz2"
            files = {}
            for path in (shard, missing):
                body = path.read_bytes() if path.exists() else b"missing"
                files[path.name] = {
                    "size": len(body),
                    "sha1": hashlib.sha1(body).hexdigest(),
                    "url": "/example",
                }
            status = root / "dumpstatus.json"
            status.write_text(
                json.dumps({"jobs": {"articlesdump": {"status": "done", "files": files}}})
            )
            database = root / "partial.sqlite3"
            manifest = root / "partial.manifest.json"
            result = MODULE.build(
                root,
                status,
                database,
                manifest,
                max_article_chars=50_000,
                finalize=False,
            )
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["sources_completed"], 1)
            self.assertFalse(manifest.exists())


if __name__ == "__main__":
    unittest.main()
