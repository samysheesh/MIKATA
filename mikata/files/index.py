"""FileIndex (phase 12d) : l'index local de tous les fichiers, tenu à jour par l'USN.

- Construction initiale : lecture de la MFT de chaque volume NTFS (`usn.enumerate_files`),
  insérée en lots dans SQLite (WAL). Quelques secondes, jamais un parcours du disque.
- Mise à jour : le journal USN (`usn.read_journal`) rejoue créations, renommages et
  suppressions toutes les `files_index.poll_s` secondes. En cas d'erreur (journal vidé,
  volume disparu), l'index se reconstruit tout seul, au plus une fois par minute.
- Recherche : FTS5 trigram sur le nom plié — sous-chaîne insensible aux accents en
  quelques millisecondes. Le LLM ne voit jamais l'index : il interroge par outils.

Le thread dédié (« mikata-file-index ») possède la connexion en écriture ; les requêtes
ouvrent leur propre connexion en lecture (WAL). Toute erreur est un avertissement, jamais
une panne : l'index est un plus, pas une dépendance vitale.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

from loguru import logger

from mikata.config import Config
from mikata.files import events, spotlight, usn

_INSERT_BATCH = 5000
_REBUILD_MIN_INTERVAL_S = 60.0
_TRIGRAM_MIN = 3  # FTS5 trigram : pas de correspondance sous 3 caractères

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files(
  id INTEGER PRIMARY KEY,
  volume TEXT NOT NULL,
  frn INTEGER NOT NULL,
  path TEXT NOT NULL UNIQUE,
  parent TEXT NOT NULL,
  name TEXT NOT NULL,
  name_fold TEXT NOT NULL,
  mtime REAL,
  is_dir INTEGER NOT NULL DEFAULT 0,
  UNIQUE(volume, frn)
);
CREATE INDEX IF NOT EXISTS idx_files_parent ON files(parent);
CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
  name_fold, content='files', content_rowid='id', tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS files_ai AFTER INSERT ON files BEGIN
  INSERT INTO files_fts(rowid, name_fold) VALUES (new.id, new.name_fold);
END;
CREATE TRIGGER IF NOT EXISTS files_ad AFTER DELETE ON files BEGIN
  DELETE FROM files_fts WHERE rowid = old.id;
END;
CREATE TRIGGER IF NOT EXISTS files_au AFTER UPDATE OF name_fold ON files BEGIN
  DELETE FROM files_fts WHERE rowid = old.id;
  INSERT INTO files_fts(rowid, name_fold) VALUES (new.id, new.name_fold);
END;
CREATE TABLE IF NOT EXISTS renames(
  volume TEXT NOT NULL,
  frn INTEGER NOT NULL,
  old_path TEXT NOT NULL,
  old_parent TEXT NOT NULL,
  PRIMARY KEY (volume, frn)
);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def fold(text: str) -> str:
    """Forme de recherche : minuscules, sans accents (comme tools/files._fold)."""
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text


def _like_prefix(path: str) -> str:
    """`path` comme préfixe LIKE : %, _ et \\ échappés, puis le joker %."""
    escaped = path.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped + "%"


def _parent_of(path: str) -> str:
    """Chemin du parent : les dossiers finissent par « \\ »."""
    if path.endswith("\\"):
        return path.rsplit("\\", 2)[0] + "\\"
    return path.rsplit("\\", 1)[0] + "\\"


# -- backend Spotlight (darwin) -----------------------------------------------------
# `spotlight_query`, `spotlight_snapshot` et `SpotlightWatcher` vivent dans
# `mikata.files.spotlight` (mdfind + FSEvents/watchdog, imports macOS paresseux) ;
# ici, seul le branchement sur `config.files_index.backend` reste. Le consommateur
# SQLite/FTS5 est identique pour les deux moteurs.


def apply_event(
    conn: sqlite3.Connection,
    volume: str,
    frn: int,
    parent_frn: int,
    name: str,
    reason: int,
    mtime: float,
    is_dir: bool,
) -> None:
    """Applique un événement USN à l'index (fonction pure, testable).

    Les FRN n'étant uniques que par volume, chaque ligne est identifiée par (volume, frn).
    Renommage d'un dossier : RENAME_OLD retire la ligne et mémorise l'ancien chemin,
    RENAME_NEW réinsère et décale les chemins de tous les enfants (leurs FRN n'ont pas
    changé, donc NTFS n'émet rien pour eux).
    """
    if reason & usn.USN_REASON_RENAME_OLD_NAME:
        row = conn.execute(
            "SELECT path, parent FROM files WHERE volume=? AND frn=?", (volume, frn)
        ).fetchone()
        if row is not None:
            conn.execute(
                "INSERT OR REPLACE INTO renames(volume, frn, old_path, old_parent) "
                "VALUES (?, ?, ?, ?)",
                (volume, frn, row[0], row[1]),
            )
            conn.execute(
                "DELETE FROM files WHERE volume=? AND frn=?", (volume, frn)
            )  # la ligne, pas les enfants
        return
    if reason & usn.USN_REASON_FILE_DELETE:
        row = conn.execute(
            "SELECT path, is_dir FROM files WHERE volume=? AND frn=?", (volume, frn)
        ).fetchone()
        if row is not None:
            if row[1]:  # un dossier : tout son contenu part avec lui
                conn.execute(
                    "DELETE FROM files WHERE path = ? OR path LIKE ? ESCAPE '\\'",
                    (row[0], _like_prefix(row[0])),
                )
            else:
                conn.execute("DELETE FROM files WHERE volume=? AND frn=?", (volume, frn))
            conn.execute("DELETE FROM renames WHERE volume=? AND frn=?", (volume, frn))
        return
    if reason & (usn.USN_REASON_FILE_CREATE | usn.USN_REASON_RENAME_NEW_NAME):
        parent = conn.execute(
            "SELECT path FROM files WHERE volume=? AND frn=?", (volume, parent_frn)
        ).fetchone()
        old = conn.execute(
            "SELECT old_path FROM renames WHERE volume=? AND frn=?", (volume, frn)
        ).fetchone()
        if parent is None:
            if old is None:
                return  # parent pas encore indexé : la reconstruction le rattrapera
            parent_path = old[0].rsplit("\\", 2)[0] + "\\"  # le parent d'avant
        else:
            parent_path = parent[0]
        path = parent_path + name + ("\\" if is_dir else "")
        conn.execute(
            """
            INSERT INTO files(volume, frn, path, parent, name, name_fold, mtime, is_dir)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(volume, frn) DO UPDATE SET
              path=excluded.path, parent=excluded.parent, name=excluded.name,
              name_fold=excluded.name_fold, mtime=excluded.mtime, is_dir=excluded.is_dir
            """,
            (volume, frn, path, parent_path, name, fold(name), mtime, int(is_dir)),
        )
        if old is not None and is_dir and old[0] != path:
            # Les enfants gardent leur fin de chemin ; seul le préfixe change.
            children = conn.execute(
                "SELECT id, path FROM files WHERE path LIKE ? ESCAPE '\\'",
                (_like_prefix(old[0]),),
            ).fetchall()
            conn.executemany(
                """
                UPDATE files SET path=?, parent=? WHERE id=?
                """,
                [
                    (
                        path + child[1][len(old[0]) :],
                        (path + child[1][len(old[0]) :]).rsplit("\\", 2)[0] + "\\",
                        child[0],
                    )
                    for child in children
                ],
            )
        conn.execute("DELETE FROM renames WHERE volume=? AND frn=?", (volume, frn))


def search_rows(conn: sqlite3.Connection, query: str, max_results: int) -> list[dict[str, Any]]:
    """Lignes de l'index qui correspondent à `query` (trigramme, sous-chaîne)."""
    wanted = fold(query).strip()
    if len(wanted) < _TRIGRAM_MIN:
        return []
    match = '"' + wanted.replace('"', '""') + '"'
    rows = conn.execute(
        """
        SELECT path, name, is_dir, mtime FROM files
        JOIN files_fts ON files_fts.rowid = files.id
        WHERE files_fts.name_fold MATCH ?
        ORDER BY (files.name_fold = ?) DESC, (files.name_fold LIKE ?) DESC, mtime DESC
        LIMIT ?
        """,
        (match, wanted, wanted + "%", max_results),
    ).fetchall()
    return [
        {"path": row[0], "name": row[1], "is_dir": bool(row[2]), "mtime": row[3]} for row in rows
    ]


def list_rows(conn: sqlite3.Connection, folder: str, limit: int) -> list[dict[str, Any]]:
    """Contenu d'un dossier depuis l'index : dossiers d'abord, puis par nom."""
    prefix = folder.rstrip("\\") + "\\"
    rows = conn.execute(
        """
        SELECT name, is_dir, mtime FROM files
        WHERE parent = ?
        ORDER BY is_dir DESC, name_fold ASC
        LIMIT ?
        """,
        (prefix, limit),
    ).fetchall()
    return [{"name": row[0], "is_dir": bool(row[1]), "mtime": row[2]} for row in rows]


def _insert_rows(
    conn: sqlite3.Connection,
    rows: list[tuple[str, int, str, str, str, str, float | None, int]],
    commit: bool = True,
) -> None:
    conn.executemany(
        """
        INSERT INTO files(volume, frn, path, parent, name, name_fold, mtime, is_dir)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    if commit:
        conn.commit()


def import_dump(conn: sqlite3.Connection, dump: Path) -> int:
    """Remplace l'index par le contenu du dump du helper (fonction pure, testable).

    L'indexation FTS incrémentale est coupée pendant l'import (elle trigrammerait
    1,6 M de noms un par un) : la table FTS est reconstruite d'un bloc à la fin.
    """
    conn.execute("DROP TRIGGER IF EXISTS files_ai")
    conn.execute("DROP TRIGGER IF EXISTS files_ad")
    conn.execute("DROP TRIGGER IF EXISTS files_au")
    conn.execute("DELETE FROM files")
    conn.execute("DELETE FROM renames")
    conn.commit()
    total = 0
    batch: list[tuple[str, int, str, str, str, str, float | None, int]] = []
    for line in events.iter_lines(dump):
        parsed = events.parse_dump_line(line)
        if parsed is None:
            continue
        path: str = parsed["path"]
        name = path.rstrip("\\").rsplit("\\", 1)[-1]
        batch.append(
            (
                path[:3],
                parsed["frn"],
                path,
                _parent_of(path),
                name,
                fold(name),
                parsed["mtime"],
                int(parsed["is_dir"]),
            )
        )
        total += 1
        if len(batch) >= _INSERT_BATCH:
            _insert_rows(conn, batch, commit=False)
            batch = []
    _insert_rows(conn, batch, commit=False)
    conn.execute("INSERT INTO files_fts(files_fts) VALUES ('rebuild')")
    conn.commit()
    conn.executescript(_SCHEMA)  # recrée les triggers (IF NOT EXISTS)
    return total


def apply_events_file(conn: sqlite3.Connection, path: Path) -> int:
    """Applique les événements JSONL du helper puis tronque les octets consommés.

    Les lignes sans saut de ligne final (append en cours) ne sont pas touchées. Renvoie
    le nombre d'événements appliqués.
    """
    if not path.exists():
        return 0
    try:
        data = path.read_bytes()
    except OSError:
        return 0
    end = data.rfind(b"\n")
    if end < 0:
        return 0
    consumed = end + 1
    applied = 0
    for line in data[:consumed].decode("utf-8", errors="replace").splitlines():
        parsed = events.parse_event_line(line)
        if parsed is None:
            continue
        apply_event(
            conn,
            parsed["volume"],
            parsed["frn"],
            parsed["parent"],
            parsed["name"],
            parsed["reason"],
            parsed["mtime"],
            bool(parsed["is_dir"]),
        )
        applied += 1
    events.consume(path, consumed)
    return applied


class FileIndex:
    """Le service d'index (thread dédié), branché sur `ToolContext.index`.

    Le moteur suit `config.files_index.backend` : « spotlight » (mdfind + FSEvents,
    macOS) ou « usn » (accès direct aux volumes en administrateur, sinon fichiers
    produits par le helper élevé dans `data/index/`).
    """

    def __init__(self, config: Config) -> None:
        self._path = config.resolve(config.files_index.db)
        self._share_dir = self._path.parent / "index"  # build.jsonl + events.log
        self._poll_s = config.files_index.poll_s
        self._backend = config.files_index.backend  # "usn" | "spotlight"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ready = False
        self._count = 0
        self._built_at = ""
        self._volumes: list[str] = []
        self._source = "direct"
        self._last_rebuild = 0.0
        self._watcher: spotlight.SpotlightWatcher | None = None

    # -- cycle de vie (patron AmbientState) --------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="mikata-file-index", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def aclose(self) -> None:
        self.stop()

    # -- lecture (requêtes : connexion propre, WAL) ------------------------------------

    def ready(self) -> bool:
        return self._ready

    def stats(self) -> dict[str, Any]:
        return {
            "ready": self._ready,
            "entries": self._count,
            "built_at": self._built_at,
            "volumes": list(self._volumes),
        }

    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        if self._backend == "spotlight":
            # macOS : Spotlight en direct (l'index SQLite reste le cache FTS5).
            try:
                paths = spotlight.spotlight_query(query, max_results)
            except Exception as exc:
                logger.warning("Recherche Spotlight impossible : {}", exc)
                return []
            return [
                {"path": p, "name": p.rstrip("/").rsplit("/", 1)[-1], "is_dir": False, "mtime": 0.0}
                for p in paths
            ]
        if not self._ready:
            return []
        try:
            conn = sqlite3.connect(self._path)
            try:
                return search_rows(conn, query, max_results)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning("Recherche dans l'index impossible : {}", exc)
            return []

    def list_folder(self, folder: str, limit: int) -> list[dict[str, Any]]:
        if not self._ready:
            return []
        try:
            conn = sqlite3.connect(self._path)
            try:
                return list_rows(conn, folder, limit)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning("Lecture de l'index impossible : {}", exc)
            return []

    # -- thread ------------------------------------------------------------------------

    def _run(self) -> None:
        try:
            conn = sqlite3.connect(self._path, timeout=5)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            try:
                self._ensure_built(conn)
                while not self._stop.is_set():
                    self._poll_once(conn)
                    self._stop.wait(self._poll_s)
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("Index des fichiers indisponible : {}", exc)

    def _ensure_built(self, conn: sqlite3.Connection) -> None:
        try:
            conn.executescript(_SCHEMA)
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                "FTS5 trigram indisponible dans ce Python : index désactivé"
            ) from exc
        # Migration phase 12d : l'ancien schéma (frn clé primaire, FTS sans content=)
        # est remplacé par le schéma (volume, frn) + FTS en contenu externe.
        files_sql = conn.execute("SELECT sql FROM sqlite_master WHERE name='files'").fetchone()
        if files_sql is not None and "volume TEXT" not in files_sql[0]:
            conn.execute("DROP TABLE files")
            conn.execute("DROP TABLE renames")
            conn.execute("DROP TABLE files_fts")
            conn.execute("DROP TABLE meta")
            conn.executescript(_SCHEMA)
        fts_sql = conn.execute("SELECT sql FROM sqlite_master WHERE name='files_fts'").fetchone()
        if fts_sql is not None and "content=" not in fts_sql[0]:
            conn.execute("DROP TABLE files_fts")  # supprime aussi les triggers liés
            conn.executescript(_SCHEMA)
            if conn.execute("SELECT 1 FROM meta WHERE key='built_at'").fetchone():
                conn.execute("INSERT INTO files_fts(files_fts) VALUES ('rebuild')")
        built = conn.execute("SELECT value FROM meta WHERE key='built_at'").fetchone()
        if built is not None:
            # Base déjà construite : prête immédiatement, sans réimport (le flag
            # `ready` doit être levé ici, sinon l'index répond « se construit »).
            self._count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            self._built_at = built[0]
            self._source = "direct" if usn.can_read_volumes() else "helper"
            self._ready = True
            logger.info("Index des fichiers : {} entrées (base existante).", self._count)
            return
        # Source directe (MIKATA en administrateur) ou fichiers du helper élevé.
        # Backend « spotlight » : mdfind + FSEvents/watchdog, jamais NTFS.
        if self._backend == "spotlight":
            self._source = "spotlight"
            self._build_from_spotlight(conn)
            return
        if usn.can_read_volumes():
            self._source = "direct"
            self._rebuild_direct(conn)
        else:
            self._source = "helper"
            self._build_from_helper(conn)

    def _rebuild_direct(self, conn: sqlite3.Connection) -> None:
        started = time.monotonic()
        self._ready = False
        volumes = usn.list_ntfs_volumes()
        self._volumes = volumes
        conn.execute("DROP TRIGGER IF EXISTS files_ai")
        conn.execute("DROP TRIGGER IF EXISTS files_ad")
        conn.execute("DROP TRIGGER IF EXISTS files_au")
        conn.execute("DELETE FROM files")
        conn.execute("DELETE FROM renames")
        conn.execute("DELETE FROM meta")
        conn.commit()
        for volume in volumes:
            if self._stop.is_set():
                return
            journal_id, next_usn = usn.query_journal(volume)
            batch: list[tuple[str, usn.UsnEntry]] = []
            for path, entry in usn.build_paths(usn.enumerate_files(volume), volume):
                batch.append((path, entry))
                if len(batch) >= _INSERT_BATCH:
                    self._insert_batch(conn, volume, batch)
                    batch = []
                    if self._stop.is_set():
                        return
            self._insert_batch(conn, volume, batch)
            conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                (f"journal:{volume}", f"{journal_id}:{next_usn}"),
            )
            if self._stop.is_set():
                return
        built_at = time.strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('built_at', ?)", (built_at,))
        conn.execute("INSERT INTO files_fts(files_fts) VALUES ('rebuild')")
        conn.commit()
        conn.executescript(_SCHEMA)  # recrée les triggers (IF NOT EXISTS)
        self._count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        self._built_at = built_at
        self._ready = True
        logger.info(
            "Index des fichiers : {} entrées en {:.1f} s ({} volume(s) NTFS, accès direct).",
            self._count,
            time.monotonic() - started,
            len(volumes),
        )

    def _build_from_spotlight(self, conn: sqlite3.Connection) -> None:
        """Backend « spotlight » : remplit le cache SQLite/FTS5 depuis Spotlight + arme FSEvents."""
        started = time.monotonic()
        self._ready = False
        try:
            rows = spotlight.spotlight_snapshot()
        except Exception as exc:
            logger.warning("Index Spotlight indisponible : {}", exc)
            return
        conn.execute("DELETE FROM files")
        conn.execute("DELETE FROM renames")
        conn.commit()
        batch: list[tuple[str, int, str, str, str, str, float | None, int]] = []
        for i, (path, mtime) in enumerate(rows):
            name = path.rstrip("/").rsplit("/", 1)[-1]
            batch.append(("", i, path, path.rsplit("/", 1)[0] + "/", name, fold(name), mtime, 0))
            if len(batch) >= _INSERT_BATCH:
                _insert_rows(conn, batch, commit=False)
                batch = []
        _insert_rows(conn, batch, commit=False)
        conn.execute("INSERT INTO files_fts(files_fts) VALUES ('rebuild')")
        built_at = time.strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('built_at', ?)", (built_at,))
        conn.commit()
        self._count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        self._built_at = built_at
        self._ready = True
        logger.info(
            "Index des fichiers : {} entrées en {:.1f} s (Spotlight).",
            self._count,
            time.monotonic() - started,
        )
        if self._watcher is None:
            self._watcher = spotlight.SpotlightWatcher(["/Users", "/Applications"])
            self._watcher.start()

    def _build_from_helper(self, conn: sqlite3.Connection) -> None:
        """Import du dump produit par le helper élevé (utilisateur standard)."""
        dump = self._share_dir / events.DUMP
        if not dump.exists():
            logger.warning(
                "Index des fichiers en attente : le helper élevé n'a pas encore produit "
                "{} (installe-le avec scripts/install_usn_helper.ps1).",
                dump,
            )
            return
        started = time.monotonic()
        self._ready = False
        import_dump(conn, dump)
        conn.execute("DELETE FROM meta")
        built_at = time.strftime("%Y-%m-%d %H:%M:%S")
        conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('built_at', ?)", (built_at,))
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES ('helper_dump_mtime', ?)",
            (str(dump.stat().st_mtime),),
        )
        conn.commit()
        self._count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        self._built_at = built_at
        self._ready = True
        logger.info(
            "Index des fichiers : {} entrées en {:.1f} s (dump du helper élevé).",
            self._count,
            time.monotonic() - started,
        )

    def _insert_batch(
        self, conn: sqlite3.Connection, volume: str, batch: list[tuple[str, usn.UsnEntry]]
    ) -> None:
        _insert_rows(
            conn,
            [
                (
                    volume,
                    entry.frn,
                    path,
                    _parent_of(path),
                    entry.name,
                    fold(entry.name),
                    entry.mtime,
                    int(entry.is_dir),
                )
                for path, entry in batch
            ],
        )

    def _poll_once(self, conn: sqlite3.Connection) -> None:
        if self._source == "spotlight":
            return  # darwin : le direct vient de FSEvents/watchdog, le reste via mdfind
        if self._source == "helper":
            self._poll_helper(conn)
            return
        volumes = self._volumes
        changed = False
        for volume in list(volumes):
            row = conn.execute(
                "SELECT value FROM meta WHERE key=?", (f"journal:{volume}",)
            ).fetchone()
            if row is None:
                continue
            try:
                journal_id, start = row[0].split(":", 1)
                events, next_usn = usn.read_journal(
                    volume, int(start), int(journal_id), timeout_s=0.0
                )
            except Exception as exc:
                logger.debug("Journal USN de {} illisible ({}).", volume, exc)
                if time.monotonic() - self._last_rebuild > _REBUILD_MIN_INTERVAL_S:
                    self._last_rebuild = time.monotonic()
                    self._rebuild_direct(conn)
                return
            for event in events:
                if usn.relevant(event.reason):
                    apply_event(
                        conn,
                        volume,
                        event.frn,
                        event.parent_frn,
                        event.name,
                        event.reason,
                        event.mtime,
                        event.is_dir,
                    )
                    changed = True
            conn.execute(
                "UPDATE meta SET value=? WHERE key=?",
                (f"{journal_id}:{next_usn}", f"journal:{volume}"),
            )
        if changed:
            conn.commit()
            self._count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]

    def _poll_helper(self, conn: sqlite3.Connection) -> None:
        """Mode utilisateur standard : consommation du flux du helper élevé."""
        if not self._ready:
            self._build_from_helper(conn)  # le helper a pu démarrer depuis
            if not self._ready:
                return
        dump = self._share_dir / events.DUMP
        try:
            dump_mtime = dump.stat().st_mtime
        except OSError:
            return
        row = conn.execute("SELECT value FROM meta WHERE key='helper_dump_mtime'").fetchone()
        if row is None or row[0] != str(dump_mtime):
            # Le helper a régénéré le dump (erreur de journal) : on repart de zéro.
            self._build_from_helper(conn)
            return
        applied = apply_events_file(conn, self._share_dir / events.EVENTS)
        if applied:
            conn.commit()
            self._count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
