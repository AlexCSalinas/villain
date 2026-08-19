"""Export and re-import: the history survives a trip between machines."""


import pytest

from villain.db import Store
from villain.portable import UnreadableExport, export_hands, import_export, read_export


def _store(tmp_path, name="a.db"):
    return Store(tmp_path / name)


def _hands():
    from tests.conftest import FIXTURE
    from villain.parsers import parse_file
    return parse_file(FIXTURE)


def test_a_round_trip_preserves_every_hand(tmp_path):
    hands = _hands()
    with _store(tmp_path) as src:
        src.add_hands(hands)
        before = {r["hand_id"] for r in src.conn.execute("SELECT hand_id FROM hands")}
        report = export_hands(src, tmp_path / "out.villain.gz")
    assert report.hands == len(before)

    with _store(tmp_path, "b.db") as dst:
        import_export(dst, tmp_path / "out.villain.gz")
        after = {r["hand_id"] for r in dst.conn.execute("SELECT hand_id FROM hands")}
    assert after == before


def test_importing_the_same_archive_twice_adds_nothing(tmp_path):
    with _store(tmp_path) as src:
        src.add_hands(_hands())
        export_hands(src, tmp_path / "out.villain.gz")
    with _store(tmp_path, "b.db") as dst:
        first = import_export(dst, tmp_path / "out.villain.gz")
        second = import_export(dst, tmp_path / "out.villain.gz")
    assert first.hands_new > 0
    assert second.hands_new == 0
    assert second.duplicates == first.hands_new


def test_a_merge_leaves_books_built(tmp_path):
    """The point of importing is a usable database, not just stored rows."""
    with _store(tmp_path) as src:
        src.add_hands(_hands())
        export_hands(src, tmp_path / "out.villain.gz")
    with _store(tmp_path, "b.db") as dst:
        import_export(dst, tmp_path / "out.villain.gz")
        assert dst.books_missing() == 0
        assert dst.conn.execute("SELECT COUNT(*) c FROM books").fetchone()["c"] > 0


def test_a_file_that_is_not_an_export_is_refused(tmp_path):
    junk = tmp_path / "junk.gz"
    import gzip
    with gzip.open(junk, "wt") as fh:
        fh.write("not json at all\n")
    with pytest.raises(UnreadableExport):
        read_export(junk)


def test_an_export_from_a_newer_villain_is_refused(tmp_path):
    import gzip
    import json
    newer = tmp_path / "newer.gz"
    with gzip.open(newer, "wt") as fh:
        fh.write(json.dumps({"villain_export": 99}) + "\n")
    with pytest.raises(UnreadableExport, match="newer villain"):
        read_export(newer)
