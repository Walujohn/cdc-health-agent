"""
rag.py: RAG utilities for chunking, embedding, building vector DBs (FAISS),
and retrieving relevant chunks (single or multi-DB).
"""

import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings

def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks for embeddings/search."""
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += chunk_size - overlap
    return chunks

def embed_chunks(chunks, embedding_model):
    """Get embeddings for each chunk using the provided embedding model."""
    return embedding_model.embed_documents(chunks)

def build_faiss_index(chunks, embeddings):
    """Build a FAISS index from text chunks and their embeddings."""
    d = len(embeddings[0])
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings).astype('float32'))
    return index

def retrieve_relevant_chunks(query, chunks, index, embedding_model, top_k=3):
    """Return top_k most relevant text chunks for the query."""
    query_vec = np.array(embedding_model.embed_query(query)).astype('float32').reshape(1, -1)
    D, I = index.search(query_vec, top_k)
    return [chunks[i] for i in I[0]]

def multi_vector_retrieve(query, dbs, embedding_model, top_k=3):
    all_results = []
    for db in dbs:
        chunks = db["chunks"]
        index = db["index"]
        D, I = index.search(
            np.array(embedding_model.embed_query(query)).astype('float32').reshape(1, -1),
            top_k
        )
        results = [(chunks[i], D[0][j]) for j, i in enumerate(I[0])]
        all_results.extend(results)
    # Sort by similarity (lowest distance first)
    all_results.sort(key=lambda x: x[1])
    return [r[0] for r in all_results[:top_k]]
