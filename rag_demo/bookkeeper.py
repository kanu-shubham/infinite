"""
bookkeeper.py — The "memory" of the indexing pipeline.

In production this is PostgreSQL. Here we use SQLite (same SQL, no server needed).

Three tables:
    document_chunks      → which chunks belong to which document
    document_permissions → which permission tokens does each document require
    query_log            → audit trail of every search (for compliance / debugging)

Why do we need this?
    When a document is updated, we need to:
    1. Find all its old chunk IDs (to delete from vector store and BM25)
    2. Insert the new chunks
    3. Update permissions if they changed

    Without the bookkeeper we'd have no way to find the old chunks —
    the vector store only lets you search by embedding, not by doc_id.
"""
import sqlite3
import json
from datetime import datetime, timezone


class Bookkeeper:

    def __init__(self, db_path: str = ":memory:"):
        # ":memory:" = in-RAM SQLite — great for tests and demos.
        # For production: db_path = "/var/lib/rag/bookkeeper.db"
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #

    def _create_tables(self) -> None:
        cursor = self.conn.cursor()

        # Table 1: document_chunks
        # Each row = one chunk. Lets us find all chunks for a doc_id.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                chunk_id   TEXT PRIMARY KEY,
                doc_id     TEXT NOT NULL,
                doc_title  TEXT,
                doc_source TEXT,
                chunk_index INTEGER,
                content_preview TEXT,
                indexed_at TEXT
            )
        """)

        # Table 2: document_permissions
        # Each row = one doc → one permission token.
        # A doc with 2 tokens has 2 rows.
        # This is the "one-to-many" relationship — normalised form.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_permissions (
                doc_id     TEXT NOT NULL,
                permission TEXT NOT NULL,
                PRIMARY KEY (doc_id, permission)
            )
        """)

        # Table 3: query_log
        # Audit trail — who searched what, when, how many results.
        # Required for GDPR, SOC2, enterprise compliance.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT,
                query      TEXT,
                num_results INTEGER,
                pattern    TEXT,
                queried_at TEXT
            )
        """)

        # Index makes "WHERE doc_id = ?" fast even with millions of rows
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
            ON document_chunks(doc_id)
        """)

        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Write operations
    # ------------------------------------------------------------------ #

    def register_chunks(self, doc_id: str, chunks) -> None:
        """
        Record that these chunks now belong to this document.
        Called after a document is indexed for the first time.
        """
        cursor = self.conn.cursor()
        now = datetime.now(timezone.utc).isoformat()

        for chunk in chunks:
            cursor.execute("""
                INSERT OR REPLACE INTO document_chunks
                (chunk_id, doc_id, doc_title, doc_source, chunk_index, content_preview, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.chunk_id,
                doc_id,
                chunk.doc_title,
                chunk.doc_source,
                chunk.chunk_index,
                chunk.content[:100],   # first 100 chars for debugging
                now,
            ))
        self.conn.commit()

    def set_permissions(self, doc_id: str, permissions: set[str]) -> None:
        """
        Replace the permission tokens for a document.
        Old tokens are deleted first, then new ones inserted.
        This handles the case where permissions change on update.
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM document_permissions WHERE doc_id = ?", (doc_id,))
        for perm in permissions:
            cursor.execute(
                "INSERT INTO document_permissions (doc_id, permission) VALUES (?, ?)",
                (doc_id, perm),
            )
        self.conn.commit()

    def log_query(self, user_id: str, query: str, num_results: int, pattern: str) -> None:
        """
        Audit log — record every search.
        In production: also store latency, model version, query embedding hash.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO query_log (user_id, query, num_results, pattern, queried_at)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, query, num_results, pattern, datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    # ------------------------------------------------------------------ #
    # Read operations
    # ------------------------------------------------------------------ #

    def get_chunk_ids_for_doc(self, doc_id: str) -> list[str]:
        """
        Return all chunk_ids that belong to this document.
        Used during upsert: delete old chunks before inserting new ones.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT chunk_id FROM document_chunks WHERE doc_id = ?", (doc_id,)
        )
        return [row[0] for row in cursor.fetchall()]

    def get_permissions(self, doc_id: str) -> set[str]:
        """Return the permission tokens required to read this document."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT permission FROM document_permissions WHERE doc_id = ?", (doc_id,)
        )
        return {row[0] for row in cursor.fetchall()}

    def delete_document(self, doc_id: str) -> list[str]:
        """
        Remove all bookkeeper records for a document.
        Returns the chunk_ids so the caller can delete them from the stores.
        """
        chunk_ids = self.get_chunk_ids_for_doc(doc_id)
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM document_chunks WHERE doc_id = ?", (doc_id,))
        cursor.execute("DELETE FROM document_permissions WHERE doc_id = ?", (doc_id,))
        self.conn.commit()
        return chunk_ids
