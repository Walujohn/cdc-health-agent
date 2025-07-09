from langgraph.graph import StateGraph
from typing_extensions import TypedDict
import os

from cdc_agent.tools.cdc_scraper import fetch_cdc_guidance
from cdc_agent.rag import chunk_text, embed_chunks, build_faiss_index, retrieve_relevant_chunks

from langchain_openai import OpenAI, OpenAIEmbeddings

# Define your agent state schema
class AgentState(TypedDict):
    question: str
    topic: str
    cdc_info: str
    summary: str

# Node: plan topic based on question
def plan_node(state: AgentState) -> AgentState:
    q = state["question"].lower()
    topic = "flu" if "flu" in q else ("monkeypox" if "monkeypox" in q else "covid")
    return {"topic": topic}

# Node: fetch CDC info
def fetch_node(state: AgentState) -> AgentState:
    info = fetch_cdc_guidance(state["topic"])
    return {"cdc_info": info}

# Node: summarize with LLM
def summarize_node(state: AgentState) -> AgentState:
    # RAG step: embed and retrieve
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


def main():
    print("📡 CDC Health Agent (LangGraph demo)")
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

if __name__ == "__main__":
    main()

