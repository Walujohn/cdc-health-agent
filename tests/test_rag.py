import numpy as np
import pytest
import asyncio

from cdc_agent.rag import (
    chunk_text,
    build_faiss_index,
    retrieve_relevant_chunks,
    multi_vector_retrieve,
    async_multi_topic_rag,
)

class DummyEmbeddings:
    def embed_documents(self, xs):
        return [np.ones(3) * i for i in range(len(xs))]
    def embed_query(self, q):
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

@pytest.mark.asyncio
async def test_async_multi_topic_rag_basic(monkeypatch):
    from cdc_agent import rag

    async def fake_fetch_cdc_guidance_async(topic, max_depth=1):
        return "Here is some CDC.gov test content about " + topic

    def fake_embed_chunks(chunks, embedding_model):
        return [np.ones(3)] * len(chunks)

    def fake_build_faiss_index(chunks, embeddings):
        class DummyIndex:
            def search(self, query_vec, top_k):
                return np.array([[0.0] * top_k]), np.array([list(range(top_k))])
        return DummyIndex()

    def fake_multi_vector_retrieve(query, dbs, embedding_model, top_k=3):
        return ["fake chunk 1", "fake chunk 2"]

    # Proper DummyAsyncOpenAI class for streaming
    class DummyStream:
        def __init__(self):
            self.done = False
        def __aiter__(self):
            return self
        async def __anext__(self):
            if not self.done:
                self.done = True
                # Simulate a real OpenAI streaming chunk
                return type(
                    "Chunk", (), {
                        "choices": [
                            type("C", (), {
                                "delta": type("D", (), {"content": "Summary."})()
                            })
                        ]
                    }
                )()
            else:
                raise StopAsyncIteration()

    class DummyAsyncOpenAI:
        def __init__(self, api_key=None):
            pass
        class chat:
            class completions:
                @staticmethod
                async def create(*args, **kwargs):
                    return DummyStream()

    monkeypatch.setattr(rag, "fetch_cdc_guidance_async", fake_fetch_cdc_guidance_async)
    monkeypatch.setattr(rag, "embed_chunks", fake_embed_chunks)
    monkeypatch.setattr(rag, "build_faiss_index", fake_build_faiss_index)
    monkeypatch.setattr(rag, "multi_vector_retrieve", fake_multi_vector_retrieve)
    monkeypatch.setattr(rag, "OPENAI_API_KEY", "dummy")
    monkeypatch.setattr(rag, "openai", type("openai", (), {"AsyncOpenAI": DummyAsyncOpenAI}))

    result = await rag.async_multi_topic_rag("What is flu?")
    assert isinstance(result, str) or result is not None


