import mlflow
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
import os
import numpy as np

from cdc_agent.tools.cdc_scraper import fetch_cdc_guidance
from cdc_agent.rag import chunk_text, embed_chunks, build_faiss_index, multi_vector_retrieve

from cdc_agent.llm import get_llm_provider

from langchain_openai import OpenAIEmbeddings

class AgentState(TypedDict):
    question: str
    topic: str
    cdc_info: str
    summary: str

# def apply_guardrails(text):
#     """
#     Stub for LLM output guardrails.
#     In production, integrate policy checks here (e.g., with Nemo Guardrails, LangChain OutputParser, or custom code).
#     For demo, redacts the word 'password' and email addresses as a placeholder.
#     """
#     import re
#     # Redact email addresses (very basic example)
#     text = re.sub(r'\b[\w.-]+?@\w+?\.\w+?\b', '[REDACTED EMAIL]', text)
#     # Redact forbidden keywords (demo)
#     text = text.replace("password", "[REDACTED]")
#     return text

# Usage: after LLM response
# summary = llm.invoke(prompt)
# summary = apply_guardrails(summary)

def plan_node(state: AgentState) -> AgentState:
    q = state["question"].lower()
    print(f"[LOG] Planning topic for question: {q}")
    if "flu" in q:
        topic = "flu"
    elif "monkeypox" in q or "mpox" in q:
        topic = "monkeypox"
    else:
        topic = "covid"
    print(f"[LOG] Topic chosen: {topic}")
    return {"topic": topic}

def fetch_all_cdc_guidance(topics, max_depth=1):
    topic_to_content = {}
    for topic in topics:
        print(f"[LOG] Fetching and chunking for {topic}")
        text = fetch_cdc_guidance(topic, max_depth=max_depth)
        topic_to_content[topic] = text
    return topic_to_content

def build_all_vector_dbs(topic_to_content, embedding_model):
    dbs = []
    for topic, text in topic_to_content.items():
        print(f"[LOG] Building vector DB for {topic}...")
        chunks = chunk_text(text)
        if not chunks:
            print(f"[WARN] No chunks found for topic: {topic}. Skipping.")
            continue
        embeddings = embed_chunks(chunks, embedding_model)
        if not embeddings:
            print(f"[WARN] No embeddings for topic: {topic}. Skipping.")
            continue
        index = build_faiss_index(chunks, embeddings)
        dbs.append({"topic": topic, "chunks": chunks, "index": index})
    return dbs

def summarize_node(state: AgentState, dbs, embedding_model) -> AgentState:
    print(f"[LOG] Running multi-DB RAG + LLM summarization for: {state['question']}")
    # Search all vector DBs and get the most relevant chunks
    relevant_chunks = multi_vector_retrieve(state["question"], dbs, embedding_model, top_k=3)
    print(f"[LOG] Retrieved {len(relevant_chunks)} relevant chunks from all DBs.")
    context = "\n".join(relevant_chunks)

    llm = get_llm_provider()
    prompt = f"""Here is some CDC.gov content:\n{context}\n\nAnswer this question as simply and accurately as possible:\n{state['question']}"""
    print(f"[LOG] LLM prompt sent:\n{'-'*40}\n{prompt[:400]}...\n{'-'*40}")
    summary_msg = llm.invoke(prompt)
    summary_text = summary_msg.content if hasattr(summary_msg, "content") else str(summary_msg)
    print(f"[LOG] LLM summary output length: {len(summary_text)} chars")
    return {"summary": summary_text}

def cdc_source_url(topic: str) -> str:
    if topic == "flu":
        return "https://www.cdc.gov/flu/"
    elif topic == "monkeypox":
        return "https://www.cdc.gov/mpox/"
    else:
        return "https://www.cdc.gov/coronavirus/2019-ncov/"

def main():
    import time

    run_id = f"cdc-agent-{int(time.time())}"
    with mlflow.start_run(run_name=run_id):
        mlflow.set_tag("agent_type", "LangGraph RAG CDC multi-vector")
        mlflow.log_param("embedding_model", "OpenAIEmbeddings")
        mlflow.log_param("llm_model", os.getenv("LLM_PROVIDER", "openai"))
        mlflow.log_param("project", "cdc-agent")

        print("📡 CDC Health Agent (LangGraph + Multi-DB RAG demo + MLflow)")
        print("Topics supported: COVID, flu, mpox (monkeypox).")
        question = input("Ask a CDC-related health question: ")

        # At startup, build all vector DBs in-memory
        topics = ["covid", "flu", "monkeypox"]
        embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        topic_to_content = fetch_all_cdc_guidance(topics, max_depth=1)
        dbs = build_all_vector_dbs(topic_to_content, embedding_model)

        # Build the StateGraph
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("plan", plan_node)

        # To pass dbs/embedding_model, wrap summarize_node as a lambda:
        graph_builder.add_node("summarize", lambda state: summarize_node(state, dbs, embedding_model))
        graph_builder.set_entry_point("plan")
        graph_builder.add_edge("plan", "summarize")

        graph = graph_builder.compile()
        initial_state = AgentState(question=question, topic="", cdc_info="", summary="")
        final_state = graph.invoke(initial_state)

        # Log info for the question/topic/chunks/summary
        mlflow.log_param("question", question)
        mlflow.log_param("topic", final_state.get("topic", ""))
        mlflow.log_metric("summary_length", len(final_state["summary"]))
        mlflow.log_text(final_state["summary"], "summary.txt")
        mlflow.log_text(final_state["cdc_info"], "cdc_info.txt")

        print("\n--- CDC Guidance Summary ---\n")
        print(final_state["summary"])
        print("\n(Full Text Preview)\n", final_state["cdc_info"][:500], "...")
        print(f"\n(CDC source: {cdc_source_url(final_state.get('topic','covid'))})\n")

if __name__ == "__main__":
    main()

# Optional: Vertex AI embeddings stub
# from vertexai.language_models import TextEmbeddingModel
# model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
# embeddings = model.get_embeddings(["your text here"])
