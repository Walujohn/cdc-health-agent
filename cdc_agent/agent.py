import mlflow
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
import os

from cdc_agent.tools.cdc_scraper import fetch_cdc_guidance
from cdc_agent.rag import chunk_text, embed_chunks, build_faiss_index, retrieve_relevant_chunks

from langchain_openai import OpenAI, OpenAIEmbeddings

class AgentState(TypedDict):
    question: str
    topic: str
    cdc_info: str
    summary: str

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

def fetch_node(state: AgentState) -> AgentState:
    print(f"[LOG] Fetching CDC guidance for topic: {state['topic']}")
    info = fetch_cdc_guidance(state["topic"])
    if not info or "Error fetching" in info or "not found" in info:
        print(f"[WARN] Failed to fetch CDC guidance.")
        info = "Sorry, I couldn't retrieve up-to-date CDC guidance right now."
    print(f"[LOG] CDC info length: {len(info)} chars")
    return {"cdc_info": info}

def summarize_node(state: AgentState) -> AgentState:
    print(f"[LOG] Running RAG + LLM summarization for: {state['question']}")
    embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    chunks = chunk_text(state["cdc_info"])
    embeddings = embed_chunks(chunks, embedding_model)
    index = build_faiss_index(chunks, embeddings)
    relevant_chunks = retrieve_relevant_chunks(state["question"], chunks, index, embedding_model, top_k=3)
    print(f"[LOG] Retrieved {len(relevant_chunks)} relevant chunks.")

    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    context = "\n".join(relevant_chunks)
    prompt = f"""Here is some CDC.gov content:\n{context}\n\nAnswer this question as simply and accurately as possible:\n{state['question']}"""
    print(f"[LOG] LLM prompt sent:\n{'-'*40}\n{prompt[:400]}...\n{'-'*40}")
    summary = llm.invoke(prompt)
    print(f"[LOG] LLM summary output length: {len(summary)} chars")
    return {"summary": summary}

def cdc_source_url(topic: str) -> str:
    """Helper to print the likely CDC source for each topic."""
    if topic == "flu":
        return "https://www.cdc.gov/flu/"
    elif topic == "monkeypox":
        return "https://www.cdc.gov/mpox/"
    else:
        return "https://www.cdc.gov/coronavirus/2019-ncov/"

def main():
    import time
    import mlflow

    run_id = f"cdc-agent-{int(time.time())}"
    with mlflow.start_run(run_name=run_id):
        mlflow.set_tag("agent_type", "LangGraph RAG CDC")
        mlflow.log_param("embedding_model", "OpenAIEmbeddings")
        mlflow.log_param("llm_model", "OpenAI")
        mlflow.log_param("project", "cdc-agent")
        # Don't log topic/question yet—they come after execution

        print("📡 CDC Health Agent (LangGraph + RAG demo + MLflow)")
        print("Topics supported: COVID, flu, mpox (monkeypox).")
        question = input("Ask a CDC-related health question: ")

        # Build the StateGraph
        graph_builder = StateGraph(AgentState)
        graph_builder.add_node("plan", plan_node)
        graph_builder.add_node("fetch", fetch_node)
        graph_builder.add_node("summarize", summarize_node)
        graph_builder.set_entry_point("plan")
        graph_builder.add_edge("plan", "fetch")
        graph_builder.add_edge("fetch", "summarize")

        # Compile and execute
        graph = graph_builder.compile()
        initial_state = AgentState(question=question, topic="", cdc_info="", summary="")
        final_state = graph.invoke(initial_state)

        # Now log dynamic values
        mlflow.log_param("topic", final_state["topic"])
        mlflow.log_param("question", question)
        mlflow.log_metric("cdc_info_length", len(final_state["cdc_info"]))
        mlflow.log_metric("summary_length", len(final_state["summary"]))
        mlflow.log_text(final_state["summary"], "summary.txt")
        mlflow.log_text(final_state["cdc_info"], "cdc_info.txt")

        print("\n--- CDC Guidance Summary ---\n")
        print(final_state["summary"])
        print("\n(Full Text Preview)\n", final_state["cdc_info"][:500], "...")
        print(f"\n(CDC source: {cdc_source_url(final_state['topic'])})\n")

if __name__ == "__main__":
    main()

# Optional: Vertex AI embeddings stub
# from vertexai.language_models import TextEmbeddingModel
# model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
# embeddings = model.get_embeddings(["your text here"])
