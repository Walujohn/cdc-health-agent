import mlflow
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
import os
import time

from cdc_agent.config import CONFIG
from cdc_agent.utils import flatten_dict
from cdc_agent.tools.cdc_scraper import fetch_cdc_guidance
from cdc_agent.rag import chunk_text, embed_chunks, build_faiss_index, multi_vector_retrieve
from cdc_agent.llm import get_llm_provider
from langchain_openai import OpenAIEmbeddings

class AgentState(TypedDict):
    question: str
    topic: str
    cdc_info: str
    summary: str

def apply_guardrails(text):
    import re
    text = re.sub(r'\b[\w.-]+?@\w+?\.\w+?\b', '[REDACTED EMAIL]', text)
    text = text.replace("password", "[REDACTED]")
    return text

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
    relevant_chunks = multi_vector_retrieve(state["question"], dbs, embedding_model, top_k=3)
    print(f"[LOG] Retrieved {len(relevant_chunks)} relevant chunks from all DBs.")
    context = "\n".join(relevant_chunks)
    llm = get_llm_provider()
    prompt = f"""Here is some CDC.gov content:\n{context}\n\nAnswer this question as simply and accurately as possible:\n{state['question']}"""
    print(f"[LOG] LLM prompt sent:\n{'-'*40}\n{prompt[:400]}...\n{'-'*40}")
    summary_msg = llm.invoke(prompt)
    summary_text = summary_msg.content if hasattr(summary_msg, "content") else str(summary_msg)
    summary_text = apply_guardrails(summary_text)
    print(f"[LOG] LLM summary output length: {len(summary_text)} chars")
    return {"summary": summary_text, "cdc_info": context}

def cdc_source_url(topic: str) -> str:
    return CONFIG["cdc_urls"].get(topic, CONFIG["cdc_urls"]["covid"])

def format_for_accessibility(text):
    if len(text) > 300 and any(w in text.lower() for w in ["symptoms", "steps", "guidelines"]):
        bullets = [line.strip() for line in text.split(".") if line.strip()]
        return "\n- " + "\n- ".join(bullets)
    return text

def main():
    run_id = f"cdc-agent-{int(time.time())}"
    with mlflow.start_run(run_name=run_id):
        mlflow.log_params(flatten_dict(CONFIG))  # Log ALL config values automatically

        print("📡 CDC Health Agent (LangGraph + Multi-DB RAG demo + MLflow)")
        print("Topics supported:", ", ".join(CONFIG["cdc_urls"].keys()))
        question = input("Ask a CDC-related health question: ")

        topics = list(CONFIG["cdc_urls"].keys())
        embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        topic_to_content = fetch_all_cdc_guidance(topics, max_depth=1)
        dbs = build_all_vector_dbs(topic_to_content, embedding_model)

        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("plan", plan_node)
        graph_builder.add_node("summarize", lambda state: summarize_node(state, dbs, embedding_model))
        graph_builder.set_entry_point("plan")
        graph_builder.add_edge("plan", "summarize")

        graph = graph_builder.compile()
        initial_state = AgentState(question=question, topic="", cdc_info="", summary="")
        final_state = graph.invoke(initial_state)

        mlflow.log_param("question", question)
        mlflow.log_param("topic", final_state.get("topic", ""))
        mlflow.log_metric("summary_length", len(final_state["summary"]))
        mlflow.log_text(final_state["summary"], "summary.txt")
        mlflow.log_text(final_state["cdc_info"], "cdc_info.txt")

        print("\n--- CDC Guidance Summary ---\n")
        accessible_summary = format_for_accessibility(final_state["summary"])
        print(accessible_summary)
        print("\n(Full Text Preview)\n", final_state["cdc_info"][:500], "...")
        print(f"\n(CDC source: {cdc_source_url(final_state.get('topic','covid'))})\n")

if __name__ == "__main__":
    main()

# Optional: Vertex AI embeddings stub
# from vertexai.language_models import TextEmbeddingModel
# model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
# embeddings = model.get_embeddings(["your text here"])
