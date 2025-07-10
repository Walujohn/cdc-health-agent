import numpy as np
from cdc_agent.rag import chunk_text, build_faiss_index, retrieve_relevant_chunks, multi_vector_retrieve

class DummyEmbeddings:
    def embed_documents(self, xs):
        # Always return a vector of ones with a fixed size for each input
        return [np.ones(3) * i for i in range(len(xs))]
    def embed_query(self, q):
        # Always return a fixed vector
        return np.zeros(3)

def test_chunk_text_basic():
    text = "A" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) == 3
    assert chunks[0].startswith("A")
    assert chunks[1].startswith("A")

def test_embed_chunks_shape():
    text = "test embedding " * 40
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    embedding_model = DummyEmbeddings()
    embeddings = embedding_model.embed_documents(chunks)
    assert len(embeddings) == len(chunks)
    assert all(e.shape == (3,) for e in embeddings)

def test_build_faiss_index_and_retrieve():
    chunks = ["chunk one", "chunk two", "chunk three"]
    embedding_model = DummyEmbeddings()
    embeddings = embedding_model.embed_documents(chunks)
    index = build_faiss_index(chunks, embeddings)
    result = retrieve_relevant_chunks("test", chunks, index, embedding_model, top_k=2)
    assert len(result) == 2

def test_multi_vector_retrieve_merges_results():
    embedding_model = DummyEmbeddings()
    db1 = {
        "chunks": ["db1c1", "db1c2"],
        "index": build_faiss_index(["db1c1", "db1c2"], embedding_model.embed_documents(["db1c1", "db1c2"]))
    }
    db2 = {
        "chunks": ["db2c1", "db2c2"],
        "index": build_faiss_index(["db2c1", "db2c2"], embedding_model.embed_documents(["db2c1", "db2c2"]))
    }
    results = multi_vector_retrieve("q", [db1, db2], embedding_model, top_k=3)
    assert len(results) == 3
