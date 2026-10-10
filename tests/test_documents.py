from pathlib import Path

import httpx
import pytest

from multi_assistant.config import Settings
from multi_assistant.documents import OllamaEmbeddings, extract_chunks, vector_literal


def write_pdf(path: Path, text: str):
    """A small original PDF fixture; no external dataset is needed."""
    stream = f"BT /F1 12 Tf 40 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
    start = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF".encode()
    )
    path.write_bytes(data)


def test_pdf_chunking_retains_page_and_complete_text(tmp_path):
    path = tmp_path / "original.pdf"
    text = "Alpha beta gamma. " * 100
    write_pdf(path, text)
    chunks = extract_chunks(path, size=100, overlap=10)
    assert len(chunks) > 1
    assert all(chunk.page == 1 for chunk in chunks)
    rebuilt = chunks[0].text + "".join(c.text[10:] for c in chunks[1:])
    assert rebuilt == " ".join(text.split())


def test_empty_pdf_and_invalid_overlap_fail_clearly(tmp_path):
    path = tmp_path / "empty.pdf"
    write_pdf(path, "")
    with pytest.raises(ValueError, match="no extractable text"):
        extract_chunks(path)
    with pytest.raises(ValueError, match="overlap"):
        extract_chunks(path, size=10, overlap=10)


def test_embedding_dimensions_and_count_are_validated(monkeypatch):
    def wrong_dimensions(*args, **kwargs):
        return httpx.Response(
            200,
            json={"embeddings": [[1, 2]]},
            request=httpx.Request("POST", "http://localhost/api/embed"),
        )

    monkeypatch.setattr(httpx, "post", wrong_dimensions)
    with pytest.raises(ValueError, match="dimensions"):
        OllamaEmbeddings(Settings()).embed(["text"])


@pytest.mark.parametrize("vector", [[], [float("nan")], [float("inf")]])
def test_invalid_vectors_are_rejected(vector):
    with pytest.raises(ValueError):
        vector_literal(vector)
