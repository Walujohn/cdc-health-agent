import gradio as gr
import openai
import httpx
from bs4 import BeautifulSoup
import os
import asyncio
import numpy as np
import faiss
from langchain_openai import OpenAIEmbeddings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def detect_topics(question):
    q = question.lower()
    topics = []
    if "covid" in q or "coronavirus" in q:
        topics.append("covid")
    if "flu" in q or "influenza" in q:
        topics.append("flu")
    if "monkeypox" in q or "mpox" in q:
        topics.append("monkeypox")
    if not topics:
        topics = ["covid"]
    return topics

async def fetch_cdc_guidance_async(topic, max_depth=1):
    topic_urls = {
        "covid": "https://www.cdc.gov/coronavirus/2019-ncov/index.html",
        "flu": "https://www.cdc.gov/flu/index.htm",
        "monkeypox": "https://www.cdc.gov/mpox/index.html"
    }
    url = topic_urls.get(topic, "https://www.cdc.gov/")
    return await fetch_links_and_content_async(url, max_depth)

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

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += chunk_size - overlap
    return chunks

async def async_embed_chunks(chunks, embedding_model):
    return embedding_model.embed_documents(chunks)

def build_faiss_index(chunks, embeddings):
    d = len(embeddings[0])
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings).astype('float32'))
    return index

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
    all_results.sort(key=lambda x: x[1])
    return [r[0] for r in all_results[:top_k]]

async def async_rag_streaming_multi(question):
    if not OPENAI_API_KEY:
        yield "ERROR: Set your OPENAI_API_KEY environment variable."
        return

    topics = detect_topics(question)
    fetch_tasks = [fetch_cdc_guidance_async(topic, max_depth=1) for topic in topics]
    topic_texts = await asyncio.gather(*fetch_tasks)

    embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    dbs = []
    for topic, text in zip(topics, topic_texts):
        chunks = chunk_text(text)
        if not chunks:
            continue
        embeddings = await async_embed_chunks(chunks, embedding_model)
        if not embeddings:
            continue
        index = build_faiss_index(chunks, embeddings)
        dbs.append({"topic": topic, "chunks": chunks, "index": index})
    if not dbs:
        yield "Sorry, couldn't fetch CDC.gov content for those topics."
        return

    relevant_chunks = multi_vector_retrieve(question, dbs, embedding_model, top_k=3)
    context = "\n".join(relevant_chunks)
    prompt = f"""Here is some CDC.gov content:\n{context}\n\nAnswer this question as simply and accurately as possible:\n{question}"""

    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    try:
        stream = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        answer = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            answer += delta
            yield answer
    except Exception as e:
        yield f"Error with OpenAI API: {e}"

iface = gr.Interface(
    fn=async_rag_streaming_multi,
    inputs=gr.Textbox(label="Ask a CDC health question"),
    outputs=gr.Textbox(label="Response"),
    title="CDC Health Agent (Async Multi-Topic RAG + Streaming Demo)",
    description="Type a question (e.g., 'What are the symptoms of covid and flu?'). Async CDC.gov fetch for each topic, vector search, and OpenAI streaming answer.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()







