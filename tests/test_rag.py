from cdc_agent.rag import chunk_text

def test_chunk_text_basic():
    text = "A" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    # Should create three chunks with overlap
    assert len(chunks) == 3
    assert chunks[0].startswith("A")
    assert chunks[1].startswith("A")
