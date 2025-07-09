# CDC Health Advisory Agent

A proof-of-concept Python agentic app for retrieving and summarizing authoritative CDC health guidance using LangGraph, RAG, and tool integration.

## Features
- Multi-step agent orchestration with LangGraph
- CDC.gov data retrieval and summarization
- Retrieval-Augmented Generation (RAG)
- Ready for local debugging and extension

## Usage
1. Clone this repo and set up a Python 3.11+ virtual environment.
2. Install dependencies.
3. Run the agent: python -m cdc_agent.agent
4. Sample questions:

“What are the symptoms of flu?”

“What does the CDC say about mpox vaccines?”

“How is COVID-19 spread?”

## For interviewers:  
See `cdc_agent/agent.py` for workflow logic, and `cdc_agent/tools/cdc_scraper.py` for the CDC.gov tool.

