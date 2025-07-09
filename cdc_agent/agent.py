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
    if "flu" in q:
        topic = "flu"
    elif "monkeypox" in q or "mpox" in q:
        topic = "monkeypox"
    else:
        topic = "covid"
    return {"topic": topic}

def fetch_node(state: AgentState) -> AgentState:
    info = fetch_cdc_guidance(state["topic"])
    # Graceful error handling if info is missing or error message present
    if not info or "Error fetching" in info or "not found" in info:
        info = "Sorry, I couldn't retrieve up-to-date CDC guidance right now."
    return {"cdc_info": info}

def summarize_node(state: AgentState) -> AgentState:
    # RAG step: chunk, embed, index, retrieve
    embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    chunks = chunk_text(state["cdc_info"])
    embeddings = embed_chunks(chunks, embedding_model)
    index = build_faiss_index(chunks, embeddings)
    relevant_chunks = retrieve_relevant_chunks(state["question"], chunks, index, embedding_model, top_k=3)

    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    context = "\n".join(relevant_chunks)
    prompt = f"""Here is some CDC.gov content:\n{context}\n\nAnswer this question as simply and accurately as possible:\n{state['question']}"""
    summary = llm.invoke(prompt)
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
    print("📡 CDC Health Agent (LangGraph + RAG demo)")
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

    print("\n--- CDC Guidance Summary ---\n")
    print(final_state["summary"])
    print("\n(Full Text Preview)\n", final_state["cdc_info"][:500], "...")
    print(f"\n(CDC source: {cdc_source_url(final_state['topic'])})\n")

if __name__ == "__main__":
    main()

