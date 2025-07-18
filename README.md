CDC Health Advisory Agent

A proof-of-concept Python agentic app for retrieving and summarizing authoritative CDC health guidance using LangGraph, RAG, and tool integration.
Built to demonstrate agentic orchestration, RAG, MLOps, and cloud readiness (Google Vertex AI, MLflow, LangSmith) for engineering experiments and real-world production.

🚀 Features

Agentic Workflow: Multi-step orchestration with LangGraph.

CDC.gov Retrieval: Live health guidance from official CDC sources.

Retrieval-Augmented Generation (RAG): Semantic chunking and vector search with OpenAI (or Vertex AI ready).

Experiment Tracking: MLflow for local experiment tracking, easily swappable to Vertex AI or other platforms.

LLM Prompt Tracing: LangSmith for prompt/response and workflow logging.

Cloud/MLOps Ready: Code stubs for Google Vertex AI, Docker, and cloud deployment.

🏃‍♂️ Usage
  1. Clone and Set Up
      git clone https://github.com/Walujohn/cdc-health-agent.git
    
      cd cdc-health-agent
      
      python3 -m venv venv
      
      source venv/bin/activate
      
      pip install -r requirements.txt
  
  3. Set Environment Variables

      export OPENAI_API_KEY="sk-..." # OpenAI API key
      
      export LANGCHAIN_TRACING_V2="true"
      
      export LANGCHAIN_API_KEY="lsm_..." # LangSmith API key
      
      export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
      
      export LANGCHAIN_PROJECT="cdc-agent"
      
      export MLFLOW_TRACKING_URI="http://localhost:5000" 
      
      Tip: Add these to your .bashrc and run source ~/.bashrc.
  
  5. Start MLflow Tracking UI (optional but recommended)
  
      mlflow ui
  
      Visit http://localhost:5000 in your browser.
  
  6. Run the Agent
     
      python -m cdc_agent.agent
  
     💡 Sample Questions
      
      “What are the symptoms of flu?”
      
      “What does the CDC say about mpox vaccines?”
      
      “How is COVID-19 spread?”

☁️ Cloud/MLOps and Vertex AI Readiness

Experiment Tracking: MLflow is integrated locally; can be configured for Vertex AI Metadata or cloud MLflow.

LLM/Embeddings: Easily swap OpenAI for Vertex AI endpoints with minimal code change.

Production Deployment: Docker-ready; can be run as a Cloud Run or Vertex AI custom job/container.

Vertex AI Example Stub:

  from vertexai.language_models import TextEmbeddingModel
  
  model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")
  
  embeddings = model.get_embeddings(["your text here"])
  
  from vertexai.language_models import TextGenerationModel
  
  model = TextGenerationModel.from_pretrained("text-bison@001")
  
  response = model.predict("Summarize this CDC content:\n" + context)
  
  print(response.text)

📝 For Interviewers

See cdc_agent/agent.py for workflow logic, experiment tracking, and logging.

See cdc_agent/tools/cdc_scraper.py for the CDC.gov retrieval tool.

Architecture is MLOps/cloud-ready for enterprise deployment.

✨ Future Extensions

Vertex AI/Gemini/Anthropic endpoint support

Slack, email, or SMS notifications

Batch processing or scheduled runs

Docker Compose for agent + tracking server

👩‍💻 Accessibility & Fullstack Experience

I am a certified accessibility tester and have professional experience ensuring web applications meet WCAG and Section 508 standards.

I have extensive Ruby on Rails and fullstack development experience alongside Python and AI/ML work.

I care deeply about accessible, inclusive AI interfaces and would prioritize accessible UX, ARIA best practices, and automated/manual accessibility testing in any production deployment. The output and formatting of the agent is improved for screen readers, making summaries easier to read and compliant with accessibility standards.

See the code example below for a quick accessibility markup stub. 
Accessibility would be a core focus of any future UI. 

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CDC Health Agent – Accessible Demo</title>
</head>
<body>
  <main>
    <h1>CDC Health Agent – Accessible Demo</h1>
    <form aria-labelledby="cdc-question-label">
      <label id="cdc-question-label" for="question">
        Ask a CDC health question:
      </label>
      <input type="text" id="question" name="question" aria-required="true" />
      <button type="submit">Ask</button>
    </form>
    <section aria-live="polite">
      <!-- The agent's answer would appear here -->
      <p id="answer" tabindex="0"></p>
    </section>
  </main>
</body>
</html>
```

🧠 Model Selection Principles

This project is designed for easy model swapping and selection.
In production, I would evaluate models based on:

Cost (e.g., per-token or per-request pricing)

Latency (response time requirements)

Quality/Accuracy (“Brains”—how complex the reasoning or summarization must be)

Examples:

For fast, high-volume Q&A: OpenAI GPT-3.5 or Vertex AI Gemini Pro

For more nuanced/critical answers: GPT-4, Gemini Ultra, or a fine-tuned Llama/Claude

For compliance/data residency: Enterprise endpoints (Vertex AI, AWS Bedrock, or Azure OpenAI)

The project’s structure (see agent.py) allows swapping LLM endpoints or embedding models with minimal changes.

Testing

Run pytest tests/ to verify basic routing and utility functions.

---
**Note on Interview-Inspired Improvements**

This POC reflects direct feedback from a technical interview.  

Key enhancements address:

- Agentic link-following and navigation (multi-page CDC content scraping)
- Multi-vector database RAG (retrieves from multiple FAISS indexes)
- Pluggable LLM provider adapter (OpenAI, Vertex AI, Anthropic)
- API-first code stubs and production comments

  For details, see the implementation and comments in `cdc_agent/tools/cdc_scraper.py` and `cdc_agent/llm.py`.
---

## 🚦 LLM Guardrails & Policy Controls

- The agent can be integrated with frameworks like [NVIDIA Nemo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) or LangChain’s output parsers to enforce policy, redaction, or compliance constraints on LLM responses.
- This enables government or regulated deployments to block PII, filter unsafe outputs, enforce style/formatting, and log violations.

## 🛡️ Government Client Considerations

- **Trusted Data Sources:** The agent retrieves content exclusively from CDC.gov and .gov domains to ensure data authenticity and public trust.
- **PII Protection:** No personal or sensitive user data is collected, stored, or transmitted.
- **Audit-Ready Logging:** MLflow and LangSmith logging enable full traceability and auditability of model behavior, supporting compliance and troubleshooting.
- **Extensible for FedRAMP/ATO:** Architecture is ready to integrate with government cloud and security practices (e.g., VPC, access controls, logging).

## 🔌 Model Context Protocol (MCP) Ready

This agent is structured to support the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction), an open standard for connecting LLMs and tools. 
See `cdc_agent/mcp_stub.py` for an example integration stub. In production, this allows the agent to be used as a plug-and-play MCP server or to consume data/tools from any MCP-compliant client or server (such as Claude, Gemini, and future enterprise IDEs).

## ⚡ Performance & Caching

- The CDC scraper uses [requests-cache](https://requests-cache.readthedocs.io/) for disk-based HTTP caching. This accelerates repeat requests, avoids unnecessary load on .gov sites, and ensures reproducible, stable agent runs—critical for demos and regulated environments.
- In production, the agent would use an official API if available, but for this POC, HTTP caching + BeautifulSoup scraping provides fast, stable results.

## ✨ 2025-07-12 Updates

* **Async & Streaming Chatbot:**
  Added a Gradio-based chatbot (`gradio_rag_async_multi.py`) that supports async multi-topic CDC.gov retrieval, FAISS RAG search, and OpenAI LLM streaming—offering a fast, interactive UI.

* **FastAPI API Server:**
  Introduced a FastAPI server (`fastapi_app.py`) for programmatic access. Supports async requests, high concurrency, and simple integration with web or enterprise applications.

* **Enterprise-Ready RAG Pipeline:**
  Refactored `rag.py` to be fully async, support multi-vector DB retrieval, and include final LLM summarization. This improves scalability, reliability, and enterprise demo readiness.

🚦 MLOps Experiments & Background Jobs

MLflow Tracking: All API calls and jobs are logged to MLflow (parameters, latency, results).

Experiment with Hyperparameters: /ask and /ask/stream endpoints accept an optional chunk_size param, letting you track and compare vector search quality/latency in MLflow.

Background MLOps Jobs:

Use /run-guidance-drift to trigger a background CDC drift/check experiment, logged in MLflow.

See cdc_agent/mlops_jobs.py for modular MLOps job definitions.

To run MLflow UI locally:

mlflow ui

# Visit http://localhost:5000 in your browser

Security: All endpoints require an API key (set via env variable API_KEY).

Author
John Harris
GitHub: https://github.com/Walujohn



