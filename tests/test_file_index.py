"""Index des fichiers (phase 12d) : reconstruction des chemins, deltas USN, recherche
trigramme — les parties pures, sans Windows. Skippé si FTS5 trigram est indisponible."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mikata.files import index as fi
from mikata.files import usn


def _has_trigram() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize='trigram')")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


pytestmark = pytest.mark.skipif(not _has_trigram(), reason="FTS5 trigram indisponible")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(fi._SCHEMA)
    return conn


def _parent_of(path: str) -> str:
    if path.endswith("\\"):  # dossier : le parent est deux crans au-dessus
        return path.rsplit("\\", 2)[0] + "\\"
    return path.rsplit("\\", 1)[0] + "\\"


def _insert(
    conn: sqlite3.Connection,
    frn: int,
    path: str,
    name: str,
    is_dir: int,
    mtime: float = 0.0,
    volume: str = "D:\\",
) -> None:
    conn.execute(
        """
        INSERT INTO files(volume, frn, path, parent, name, name_fold, mtime, is_dir)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (volume, frn, path, _parent_of(path), name, fi.fold(name), mtime, is_dir),
    )


def test_build_paths_reconstructs_full_paths() -> None:
    entries = [
        usn.UsnEntry(10, usn.ROOT_FRN, "Steam", 0.0, True),
        usn.UsnEntry(11, 10, "steamapps", 0.0, True),
        usn.UsnEntry(12, 11, "Cyberpunk 2077.exe", 123.0, False),
        usn.UsnEntry(13, 99, "orphelin.txt", 0.0, False),  # parent inconnu : ignoré
        usn.UsnEntry(14, usn.ROOT_FRN, "$Extend", 0.0, True),  # sous-arbre système : ignoré
    ]
    paths = dict(usn.build_paths(entries, "D:\\"))
    assert "D:\\Steam\\steamapps\\Cyberpunk 2077.exe" in paths
    assert "D:\\Steam\\" in paths and "D:\\Steam\\steamapps\\" in paths
    assert not any("orphelin" in path for path in paths)
    assert not any("$Extend" in path for path in paths)


def test_create_event_adds_the_row() -> None:
    conn = _conn()
    _insert(conn, 10, "D:\\Steam\\", "Steam", 1)
    fi.apply_event(conn, "D:\\", 11, 10, "steamapps", usn.USN_REASON_FILE_CREATE, 5.0, True)
    row = conn.execute("SELECT path, name_fold, is_dir FROM files WHERE frn=11").fetchone()
    assert row == ("D:\\Steam\\steamapps\\", "steamapps", 1)


def test_renaming_a_folder_moves_its_children() -> None:
    conn = _conn()
    _insert(conn, 10, "D:\\A\\", "A", 1)
    _insert(conn, 11, "D:\\A\\fichier.txt", "fichier.txt", 0)
    fi.apply_event(conn, "D:\\", 10, usn.ROOT_FRN, "A", usn.USN_REASON_RENAME_OLD_NAME, 0.0, True)
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1  # l'enfant reste
    fi.apply_event(conn, "D:\\", 10, usn.ROOT_FRN, "B", usn.USN_REASON_RENAME_NEW_NAME, 0.0, True)
    rows = [r[0] for r in conn.execute("SELECT path FROM files ORDER BY path").fetchall()]
    assert rows == ["D:\\B\\", "D:\\B\\fichier.txt"]


def test_deleting_a_folder_removes_the_subtree() -> None:
    conn = _conn()
    _insert(conn, 10, "D:\\A\\", "A", 1)
    _insert(conn, 11, "D:\\A\\fichier.txt", "fichier.txt", 0)
    _insert(conn, 12, "D:\\autre.txt", "autre.txt", 0)
    fi.apply_event(conn, "D:\\", 10, usn.ROOT_FRN, "A", usn.USN_REASON_FILE_DELETE, 0.0, True)
    rows = [r[0] for r in conn.execute("SELECT path FROM files").fetchall()]
    assert rows == ["D:\\autre.txt"]


def test_same_frn_on_two_volumes_coexists() -> None:
    """Régression : les FRN sont uniques par volume, pas globalement."""
    conn = _conn()
    _insert(conn, 5, "C:\\pagefile.sys", "pagefile.sys", 0, volume="C:\\")
    _insert(conn, 5, "D:\\pagefile.sys", "pagefile.sys", 0, volume="D:\\")
    assert conn.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 2
    # Et le journal d'un volume ne touche pas l'autre.
    fi.apply_event(
        conn, "C:\\", 5, usn.ROOT_FRN, "pagefile.sys", usn.USN_REASON_FILE_DELETE, 0.0, False
    )
    rows = [r[0] for r in conn.execute("SELECT path FROM files").fetchall()]
    assert rows == ["D:\\pagefile.sys"]


def test_search_is_accent_insensitive_and_ranked() -> None:
    conn = _conn()
    _insert(conn, 1, "D:\\Docs\\Facture électricité.pdf", "Facture électricité.pdf", 0, 100.0)
    _insert(conn, 2, "D:\\Docs\\facture-eau.pdf", "facture-eau.pdf", 0, 200.0)
    _insert(conn, 3, "D:\\Docs\\notes.txt", "notes.txt", 0, 300.0)
    rows = fi.search_rows(conn, "facture", 10)
    assert [r["name"] for r in rows] == ["facture-eau.pdf", "Facture électricité.pdf"]
    rows = fi.search_rows(conn, "electricite", 10)
    assert [r["name"] for r in rows] == ["Facture électricité.pdf"]
    assert fi.search_rows(conn, "zz", 10) == []  # trigram : moins de 3 caractères, rien


def test_list_folder_directories_first() -> None:
    conn = _conn()
    _insert(conn, 1, "D:\\Docs\\", "Docs", 1)
    _insert(conn, 2, "D:\\Docs\\alpha.txt", "alpha.txt", 0)
    _insert(conn, 3, "D:\\Docs\\Sous\\", "Sous", 1)
    rows = fi.list_rows(conn, "D:\\Docs", 50)
    assert [r["name"] for r in rows] == ["Sous", "alpha.txt"]


def test_fts_stays_in_sync_through_triggers() -> None:
    conn = _conn()
    _insert(conn, 1, "D:\\Docs\\Facture électricité.pdf", "Facture électricité.pdf", 0)
    conn.commit()
    assert fi.search_rows(conn, "electricite", 10)  # trouvé via FTS
    conn.execute("DELETE FROM files WHERE frn=1")
    conn.commit()
    assert fi.search_rows(conn, "electricite", 10) == []  # retiré du FTS aussi


# -- contrat du helper élevé (JSONL) ----------------------------------------------------


def test_dump_and_event_lines_roundtrip() -> None:
    from mikata.files import events

    line = events.dump_line(
        "D:\\Steam\\Cyberpunk 2077.exe", usn.UsnEntry(12, 11, "Cyberpunk 2077.exe", 5.0, False)
    )
    parsed = events.parse_dump_line(line)
    assert parsed is not None and parsed["path"] == "D:\\Steam\\Cyberpunk 2077.exe"
    assert parsed["frn"] == 12 and parsed["is_dir"] == 0
    event_line = events.event_line(
        "D:\\", usn.UsnEvent(13, 10, "nouveau.txt", usn.USN_REASON_FILE_CREATE, 7.0, 1000, False)
    )
    event = events.parse_event_line(event_line)
    assert event is not None and event["name"] == "nouveau.txt"
    assert event["reason"] == usn.USN_REASON_FILE_CREATE and event["volume"] == "D:\\"


def test_iter_lines_ignores_a_truncated_tail(tmp_path: Path) -> None:
    from mikata.files import events

    path = tmp_path / "events.log"
    path.write_text('{"a": 1}\n{"b": 2}\n{"c": 3', encoding="utf-8")
    assert list(events.iter_lines(path)) == ['{"a": 1}', '{"b": 2}']


def test_import_dump_replaces_the_index(tmp_path: Path) -> None:
    from mikata.files import events

    dump = tmp_path / "build.jsonl"
    dump.write_text(
        "".join(
            events.dump_line(path, entry) + "\n"
            for path, entry in [
                ("D:\\Steam\\Cyberpunk 2077.exe", usn.UsnEntry(12, 11, "x", 5.0, False)),
                ("D:\\Steam\\", usn.UsnEntry(11, 10, "Steam", 5.0, True)),
            ]
        ),
        encoding="utf-8",
    )
    conn = _conn()
    _insert(conn, 99, "D:\\ancien.txt", "ancien.txt", 0)
    total = fi.import_dump(conn, dump)
    assert total == 2
    rows = {r[0] for r in conn.execute("SELECT path FROM files").fetchall()}
    assert rows == {"D:\\Steam\\Cyberpunk 2077.exe", "D:\\Steam\\"}


def test_apply_events_file_consumes_what_was_applied(tmp_path: Path) -> None:
    from mikata.files import events

    events_path = tmp_path / "events.log"
    events_path.write_text(
        events.event_line(
            "D:\\", usn.UsnEvent(20, 10, "créé.txt", usn.USN_REASON_FILE_CREATE, 1.0, 1, False)
        )
        + "\n",
        encoding="utf-8",
    )
    conn = _conn()
    _insert(conn, 10, "D:\\Docs\\", "Docs", 1)
    applied = fi.apply_events_file(conn, events_path)
    assert applied == 1
    assert conn.execute("SELECT path FROM files WHERE frn=20").fetchone()[0] == "D:\\Docs\\créé.txt"
    assert events_path.stat().st_size == 0  # consommé
    # Un append en cours (ligne sans fin) reste en place.
    events_path.write_text('{"frn": 21', encoding="utf-8")
    assert fi.apply_events_file(conn, events_path) == 0
    assert events_path.stat().st_size > 0


def test_build_from_helper_without_dump_does_not_crash(tmp_path: Path) -> None:
    """Régression : l'index référençait des constantes absentes du module des événements
    (AttributeError silencieuse dans le thread) — sans dump, il doit attendre proprement."""
    from tool_fakes import make_config

    service = fi.FileIndex(make_config(tmp_path))
    conn = _conn()
    service._build_from_helper(conn)
    assert not service.ready()
