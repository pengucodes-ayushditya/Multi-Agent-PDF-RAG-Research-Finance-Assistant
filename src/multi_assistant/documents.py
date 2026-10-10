import hashlib
import math
from pathlib import Path

import httpx
import psycopg
from psycopg import sql
from pydantic import BaseModel
from pypdf import PdfReader

from .config import Settings
from .evidence import Evidence, Source


class Chunk(BaseModel):
    text: str
    page: int
    index: int


def extract_chunks(path: Path, size: int = 1200, overlap: int = 150) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("Chunk overlap must be smaller than chunk size")
    if path.suffix.lower() != ".pdf" or not path.is_file():
        raise ValueError("Provide an existing local PDF file")
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("PDF size limit is 50 MB")
    reader = PdfReader(path)
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are unsupported")
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        for start in range(0, len(text), size - overlap):
            chunks.append(
                Chunk(text=text[start : start + size], page=page_number, index=len(chunks))
            )
            if len(chunks) > 5000:
                raise ValueError("PDF exceeds the 5,000 chunk limit")
            if start + size >= len(text):
                break
    if not chunks:
        raise ValueError("PDF has no extractable text; scanned PDFs need OCR before ingestion")
    return chunks


class OllamaEmbeddings:
    def __init__(self, settings: Settings):
        self.settings = settings

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            self.settings.ollama_base_url.rstrip("/") + "/api/embed",
            json={"model": self.settings.embedding_model, "input": texts},
            timeout=self.settings.request_timeout,
        )
        response.raise_for_status()
        vectors = response.json()["embeddings"]
        if len(vectors) != len(texts):
            raise ValueError("Embedding response count mismatch")
        for vector in vectors:
            if len(vector) != self.settings.embedding_dimensions:
                raise ValueError("Embedding dimensions differ from EMBEDDING_DIMENSIONS")
            if not all(math.isfinite(float(value)) for value in vector):
                raise ValueError("Embedding contains invalid values")
        return vectors


def vector_literal(vector: list[float]) -> str:
    if not vector or not all(math.isfinite(float(value)) for value in vector):
        raise ValueError("Invalid embedding vector")
    return "[" + ",".join(str(float(value)) for value in vector) + "]"


class DocumentStore:
    def __init__(self, settings: Settings, embeddings=None):
        self.settings = settings
        self.embeddings = embeddings or OllamaEmbeddings(settings)

    def connect(self):
        if not self.settings.database_url:
            raise ValueError("Set DATABASE_URL to use PDF retrieval")
        return psycopg.connect(self.settings.database_url.get_secret_value(), connect_timeout=5)

    def initialize(self):
        with self.connect() as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(
                sql.SQL("""
                CREATE TABLE IF NOT EXISTS assistant_chunks (
                    collection TEXT NOT NULL, document_id TEXT NOT NULL,
                    source_name TEXT NOT NULL, page INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL, content TEXT NOT NULL,
                    embedding_model TEXT NOT NULL, embedding vector({dimensions}) NOT NULL,
                    PRIMARY KEY (collection, document_id, chunk_index)
                )
            """).format(dimensions=sql.Literal(self.settings.embedding_dimensions))
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS assistant_collection_idx "
                "ON assistant_chunks(collection)"
            )
            column_type = connection.execute(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid='assistant_chunks'::regclass AND attname='embedding'"
            ).fetchone()[0]
            if column_type != f"vector({self.settings.embedding_dimensions})":
                raise ValueError(
                    "Database embedding dimensions differ from the configuration. "
                    "Use the original dimensions or a separate database."
                )

    def ingest(self, path: Path) -> int:
        chunks = extract_chunks(path)
        document_id = hashlib.sha256(path.read_bytes()).hexdigest()
        vectors = []
        for offset in range(0, len(chunks), 16):
            vectors.extend(self.embeddings.embed([c.text for c in chunks[offset : offset + 16]]))
        self.initialize()
        # A failure rolls back this document, without deleting any other documents.
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM assistant_chunks WHERE collection=%s AND document_id=%s",
                (self.settings.pdf_collection, document_id),
            )
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO assistant_chunks
                    (collection, document_id, source_name, page, chunk_index,
                     content, embedding_model, embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector)
                """,
                    [
                        (
                            self.settings.pdf_collection,
                            document_id,
                            path.name,
                            c.page,
                            c.index,
                            c.text,
                            self.settings.embedding_model,
                            vector_literal(v),
                        )
                        for c, v in zip(chunks, vectors, strict=True)
                    ],
                )
        return len(chunks)

    def search(self, query: str, limit: int = 5) -> Evidence:
        if not query.strip() or not 1 <= limit <= 10:
            raise ValueError("Provide a query and a limit between 1 and 10")
        vector = vector_literal(self.embeddings.embed([query])[0])
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT document_id, source_name, page, content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM assistant_chunks
                WHERE collection=%s AND embedding_model=%s
                ORDER BY embedding <=> %s::vector LIMIT %s
            """,
                (
                    vector,
                    self.settings.pdf_collection,
                    self.settings.embedding_model,
                    vector,
                    limit,
                ),
            ).fetchall()
        sources, excerpts = [], []
        for document_id, name, page, text, similarity in rows:
            source = Source.create(name, f"pdf:{document_id}/{name}", page)
            sources.append(source)
            excerpts.append(
                f"[{source.id}] page {page}, cosine similarity {similarity:.3f}: {text}"
            )
        return Evidence(
            data="\n\n".join(excerpts) or "No indexed PDF passages were found.", sources=sources
        )

    def inventory(self) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT document_id, source_name, count(*) FROM assistant_chunks
                WHERE collection=%s GROUP BY document_id, source_name ORDER BY source_name
            """,
                (self.settings.pdf_collection,),
            ).fetchall()
        return [dict(document_id=row[0], name=row[1], chunks=row[2]) for row in rows]
