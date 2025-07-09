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
