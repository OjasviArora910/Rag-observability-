import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.config import Settings
from app.vectorstore import VectorStore, make_client


def test_existing_collection_with_different_vector_size_uses_compatible_collection(caplog):
    client = QdrantClient(":memory:")
    client.create_collection(
        "rag_chunks",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    store = VectorStore(client, "rag_chunks", 256)

    assert store.collection == "rag_chunks_256d"
    info = client.get_collection(store.collection)
    assert info.config.params.vectors.size == 256
    assert "re-upload documents" in caplog.text

    reopened = VectorStore(client, "rag_chunks", 256)
    assert reopened.collection == "rag_chunks_256d"

    client.close()


def test_make_client_reports_existing_qdrant_lock_without_exposing_raw_traceback(tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path)
    first = make_client(settings)
    try:
        with pytest.raises(RuntimeError, match="Stop the other app process") as exc:
            make_client(settings)
        assert "already accessed by another instance" not in str(exc.value)
    finally:
        first.close()
