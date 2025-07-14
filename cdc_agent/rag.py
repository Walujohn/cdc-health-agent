"""
rag.py: Async RAG utilities for chunking, embedding, vector DBs (FAISS),
multi-topic CDC.gov retrieval, and OpenAI LLM summarization.
"""

import os
import asyncio
import faiss
import numpy as np
import httpx
from bs4 import BeautifulSoup
from langchain_openai import OpenAIEmbeddings
import openai

from cdc_agent.config import CONFIG

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def detect_topics(question):
    q = question.lower()
    topics = []
    for key in CONFIG["cdc_urls"]:
        if key in q or (key == "covid" and "coronavirus" in q):
            topics.append(key)
    if not topics:
        topics = ["covid"]
    return topics

async def fetch_links_and_content_async(start_url, max_depth=1):
    visited = set()
    queue = [(start_url, 0)]
    all_text = []
    async with httpx.AsyncClient(timeout=10) as client:
        while queue:
            url, depth = queue.pop(0)
            if url in visited or depth > max_depth:
                continue
            visited.add(url)
            try:
                resp = await client.get(url)
                soup = BeautifulSoup(resp.text, "html.parser")
                all_text.append(soup.get_text(separator=" ", strip=True))
                if depth < max_depth:
                    for a in soup.find_all("a", href=True):
                        link = a["href"]
                        if link.startswith("https://www.cdc.gov") and link not in visited:
                            queue.append((link, depth+1))
            except Exception as e:
                print(f"Failed to fetch {url}: {e}")
    return "\n".join(all_text)

async def fetch_cdc_guidance_async(topic, max_depth=1):
    url = CONFIG["cdc_urls"].get(topic, "https://www.cdc.gov/")
    return await fetch_links_and_content_async(url, max_depth)

def chunk_text(text, chunk_size=None, overlap=None, max_chunks=None):
    chunk_size = chunk_size or CONFIG["chunk_size"]
    overlap = overlap or CONFIG["overlap"]
    max_chunks = max_chunks or CONFIG["max_chunks"]
    chunks = []
    i = 0
    while i < len(text) and len(chunks) < max_chunks:
        chunk = text[i:i+chunk_size]
        if len(chunk.strip()) > 10:
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def embed_chunks(chunks, embedding_model):
    return embedding_model.embed_documents(chunks)

def build_faiss_index(chunks, embeddings):
    d = len(embeddings[0])
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings).astype('float32'))
    return index

def retrieve_relevant_chunks(query, chunks, index, embedding_model, top_k=3):
    query_vec = np.array(embedding_model.embed_query(query)).astype('float32').reshape(1, -1)
    D, I = index.search(query_vec, top_k)
    return [chunks[i] for i in I[0] if i < len(chunks)]

def multi_vector_retrieve(query, dbs, embedding_model, top_k=3):
    all_results = []
    for db in dbs:
        chunks = db["chunks"]
        index = db["index"]
        D, I = index.search(
            np.array(embedding_model.embed_query(query)).astype('float32').reshape(1, -1),
            top_k
        )
        results = [(chunks[i], D[0][j]) for j, i in enumerate(I[0]) if i < len(chunks)]
        all_results.extend(results)
    all_results.sort(key=lambda x: x[1])
    return [r[0] for r in all_results[:top_k]]

async def async_multi_topic_rag(question):
    topics = detect_topics(question)
    fetch_tasks = [fetch_cdc_guidance_async(topic, max_depth=1) for topic in topics]
    topic_texts = await asyncio.gather(*fetch_tasks)
    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    dbs = []
    for topic, text in zip(topics, topic_texts):
        chunks = chunk_text(text)
        if not chunks:
            continue
        embeddings = embed_chunks(chunks, embedding_model)
        if not embeddings:
            continue
        index = build_faiss_index(chunks, embeddings)
        dbs.append({"topic": topic, "chunks": chunks, "index": index})
    if not dbs:
        return "Sorry, couldn't fetch CDC.gov content for those topics."
    relevant_chunks = multi_vector_retrieve(question, dbs, embedding_model, top_k=3)
    context = "\n".join(relevant_chunks)
    prompt = f"""Here is some CDC.gov content:\n{context}\n\nAnswer this question as simply and accurately as possible:\n{question}"""
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    answer = ""
    stream = await client.chat.completions.create(
        model=CONFIG["llm_model"],
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        answer += delta
    return answer


