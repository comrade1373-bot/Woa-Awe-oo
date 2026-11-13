import hashlib
import os
import re
from pathlib import Path

from .core import Store, utc_now


SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".py", ".json", ".csv"}
MAX_DOCUMENT_BYTES = 5 * 1024 * 1024
MAX_BATCH_BYTES = 15 * 1024 * 1024
MAX_BATCH_FILES = 100
MAX_INDEX_BYTES = 100 * 1024 * 1024
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 160
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _chunks(text):
    start = 0
    result = []
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + CHUNK_CHARS // 2, end)
            if boundary > start:
                end = boundary
        content = text[start:end].strip()
        if content:
            result.append(content)
        if end == len(text):
            break
        start = max(start + 1, end - CHUNK_OVERLAP)
    return result


def _terms(text):
    return [word.casefold() for word in _WORD.findall(text)]


def _sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


class KnowledgeIndex:
    def __init__(self, store):
        if not isinstance(store, Store):
            raise TypeError("KnowledgeIndex requires a Store.")
        self.store = store

    def _prepare_document(self, source, text):
        if not isinstance(source, str) or not source.strip() or len(source) > 500:
            raise ValueError("A source label between 1 and 500 characters is required.")
        if Path(source).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                "Supported local knowledge formats are .txt, .md, .rst, .py, .json, and .csv."
            )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{source}: document text is empty.")
        if "\x00" in text:
            raise ValueError(f"{source}: binary content is not supported.")
        try:
            source.encode("utf-8")
            encoded = text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(f"{source}: source and text must be valid Unicode.") from error
        if len(encoded) > MAX_DOCUMENT_BYTES:
            raise ValueError(
                f"{source}: document exceeds the {MAX_DOCUMENT_BYTES}-byte limit."
            )
        pieces = _chunks(text)
        if not pieces:
            raise ValueError(f"{source}: document has no indexable text.")
        return {
            "source": source,
            "pieces": pieces,
            "content_hash": hashlib.sha256(encoded).hexdigest(),
            "indexed_at": utc_now(),
            "bytes": len(encoded),
            "chunk_bytes": sum(len(piece.encode("utf-8")) for piece in pieces),
        }

    def _store_documents(self, documents):
        sources = [document["source"] for document in documents]
        placeholders = ",".join("?" for _ in sources)
        with self.store.connect() as db:
            current_bytes = db.execute(
                "SELECT COALESCE(SUM(length(CAST(content AS BLOB))), 0) "
                "FROM knowledge_chunks"
            ).fetchone()[0]
            old_bytes = db.execute(
                "SELECT COALESCE(SUM(length(CAST(content AS BLOB))), 0) "
                f"FROM knowledge_chunks WHERE source IN ({placeholders})",
                sources,
            ).fetchone()[0]
            new_bytes = sum(document["chunk_bytes"] for document in documents)
            if current_bytes - old_bytes + new_bytes > MAX_INDEX_BYTES:
                raise ValueError(
                    f"Local knowledge index exceeds the {MAX_INDEX_BYTES}-byte limit."
                )
            results = []
            for document in documents:
                source = document["source"]
                db.execute(
                    "DELETE FROM knowledge_fts WHERE rowid IN "
                    "(SELECT rowid FROM knowledge_chunks WHERE source = ?)",
                    (source,),
                )
                db.execute("DELETE FROM knowledge_chunks WHERE source = ?", (source,))
                db.executemany(
                    "INSERT INTO knowledge_chunks "
                    "(source, content_hash, chunk_index, content, indexed_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            source,
                            document["content_hash"],
                            index,
                            piece,
                            document["indexed_at"],
                        )
                        for index, piece in enumerate(document["pieces"], start=1)
                    ],
                )
                db.executemany(
                    "INSERT INTO knowledge_fts "
                    "(rowid, source, chunk_index, content) VALUES (?, ?, ?, ?)",
                    db.execute(
                        "SELECT rowid, source, chunk_index, content "
                        "FROM knowledge_chunks WHERE source = ?",
                        (source,),
                    ).fetchall(),
                )
                results.append(
                    {
                        "source": source,
                        "chunks": len(document["pieces"]),
                        "bytes": document["bytes"],
                        "sha256": document["content_hash"],
                    }
                )
        return results

    def ingest_text(self, source, text):
        document = self._prepare_document(source, text)
        return self._store_documents([document])[0]

    def ingest_paths(self, paths):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        if not paths:
            raise ValueError("At least one file or directory path is required.")
        files = []
        for value in paths:
            path = Path(value).expanduser()
            if path.is_symlink():
                raise ValueError(f"Symbolic links are not indexed: {path}")
            try:
                resolved = path.resolve(strict=True)
            except OSError as error:
                raise ValueError(f"Could not access {path}: {error}") from error
            if resolved.is_file():
                if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    raise ValueError(
                        f"Unsupported file type: {resolved.suffix or '(no extension)'}."
                    )
                files.append(resolved)
            elif resolved.is_dir():
                def raise_walk_error(error):
                    raise error

                try:
                    for current, directories, filenames in os.walk(
                        resolved, onerror=raise_walk_error, followlinks=False
                    ):
                        directories[:] = sorted(
                            name
                            for name in directories
                            if not name.startswith(".")
                            and not (Path(current) / name).is_symlink()
                        )
                        for name in sorted(filenames):
                            candidate = Path(current) / name
                            if (
                                candidate.is_symlink()
                                or name.startswith(".")
                                or candidate.suffix.lower() not in SUPPORTED_EXTENSIONS
                            ):
                                continue
                            files.append(candidate)
                            if len(files) > MAX_BATCH_FILES:
                                raise ValueError(
                                    f"At most {MAX_BATCH_FILES} files may be indexed at once."
                                )
                except OSError as error:
                    raise ValueError(
                        f"Could not scan directory {resolved}: {error}"
                    ) from error
            else:
                raise ValueError(f"Path is not a file or directory: {resolved}")
            if len(files) > MAX_BATCH_FILES:
                raise ValueError(
                    f"At most {MAX_BATCH_FILES} files may be indexed at once."
                )
        files = list(dict.fromkeys(files))
        if not files:
            raise ValueError("No supported text or code files were found.")

        documents = []
        total_bytes = 0
        for path in files:
            try:
                with path.open("rb") as source_file:
                    encoded = source_file.read(MAX_DOCUMENT_BYTES + 1)
                if len(encoded) > MAX_DOCUMENT_BYTES:
                    raise ValueError(
                        f"{path}: document exceeds the {MAX_DOCUMENT_BYTES}-byte limit."
                    )
                total_bytes += len(encoded)
                if total_bytes > MAX_BATCH_BYTES:
                    raise ValueError(
                        f"Selected files exceed the {MAX_BATCH_BYTES}-byte batch limit."
                    )
                text = encoded.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"{path}: file is not valid UTF-8 text.") from error
            except OSError as error:
                raise ValueError(f"Could not read {path}: {error}") from error
            documents.append((str(path), text))

        prepared = [
            self._prepare_document(source, text) for source, text in documents
        ]
        return self._store_documents(prepared)

    def list_documents(self):
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT source, MAX(content_hash) AS sha256, "
                "COUNT(*) AS chunks, MAX(indexed_at) AS indexed_at "
                "FROM knowledge_chunks GROUP BY source ORDER BY source"
            ).fetchall()
        return [dict(row) for row in rows]

    def search(self, query, limit=5):
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise ValueError("A search query between 1 and 1000 characters is required.")
        try:
            query.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("Search query must be valid Unicode.") from error
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 20:
            raise ValueError("Search result limit must be an integer from 1 to 20.")
        query_terms = _terms(query)
        if not query_terms:
            raise ValueError("Search query must contain at least one word.")
        if len(query_terms) > 40:
            raise ValueError("Search query may contain at most 40 words.")

        match_expression = " OR ".join(
            f'"{term}"' for term in sorted(set(query_terms))
        )
        with self.store.connect() as db:
            rows = db.execute(
                "SELECT source, chunk_index, content, bm25(knowledge_fts) AS rank "
                "FROM knowledge_fts WHERE knowledge_fts MATCH ? "
                "ORDER BY rank, source, chunk_index LIMIT ?",
                (match_expression, limit),
            ).fetchall()
        return [
            {
                "source": row["source"],
                "chunk": row["chunk_index"],
                "score": round(-float(row["rank"]), 8),
                "content": row["content"],
            }
            for row in rows
        ]

    def digest(self, query, limit=8):
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 20:
            raise ValueError("Digest result limit must be an integer from 1 to 20.")
        hits = self.search(query, limit=20)
        query_terms = set(_terms(query))

        candidates = []
        seen = set()
        for hit in hits:
            for sentence in _sentences(hit["content"]):
                identity = (hit["source"], sentence.casefold())
                if identity in seen:
                    continue
                overlap = len(query_terms.intersection(_terms(sentence)))
                if overlap:
                    seen.add(identity)
                    candidates.append(
                        {
                            "source": hit["source"],
                            "chunk": hit["chunk"],
                            "score": round(overlap * hit["score"], 8),
                            "content": sentence,
                        }
                    )
        candidates.sort(
            key=lambda item: (-item["score"], item["source"], item["chunk"])
        )
        return candidates[:limit]
