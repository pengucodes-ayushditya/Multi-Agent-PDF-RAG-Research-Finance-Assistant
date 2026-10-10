import os
import uuid

import pytest
from test_documents import write_pdf

from multi_assistant.config import Settings
from multi_assistant.documents import DocumentStore


@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION"), reason="Set RUN_INTEGRATION=1 explicitly")
def test_real_pgvector_ingestion_retrieval_isolation_and_reingestion(tmp_path):
    settings = Settings.load().model_copy(update={"pdf_collection": "test-" + uuid.uuid4().hex})
    store = DocumentStore(settings)
    try:
        first = tmp_path / "aurora.pdf"
        second = tmp_path / "sequoia.pdf"
        write_pdf(first, "Aurora research reports revenue of 42 million dollars for 2025.")
        write_pdf(second, "Sequoia laboratory studies protein folding using microscopy.")
        assert store.ingest(first) == 1
        assert store.ingest(second) == 1
        mismatched = DocumentStore(settings.model_copy(update={"embedding_dimensions": 2}))
        with pytest.raises(ValueError, match="dimensions"):
            mismatched.initialize()
        assert store.ingest(first) == 1
        assert len(store.inventory()) == 2
        evidence = store.search("What revenue does Aurora report?", limit=1)
        assert "42 million" in evidence.data
        assert evidence.sources[0].page == 1
        assert evidence.sources[0].title == "aurora.pdf"
        other = DocumentStore(settings.model_copy(update={"pdf_collection": uuid.uuid4().hex}))
        assert not other.search("Aurora revenue").sources
    finally:
        with store.connect() as connection:
            connection.execute(
                "DELETE FROM assistant_chunks WHERE collection=%s", (settings.pdf_collection,)
            )
