import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from cdc_agent.rag import async_multi_topic_rag  # Your async RAG pipeline

app = FastAPI(
    title="CDC Agent API",
    description="Enterprise-ready async RAG (multi-topic, CDC.gov) for health Q&A",
    version="0.1.0"
)

# Optional: Enable CORS so UI/frontend can access the API anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict as needed
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_rag(req: QueryRequest):
    """
    Accepts a JSON body: {"question": "..."}
    Returns: {"answer": "..."}
    """
    try:
        # This assumes async_multi_topic_rag returns either a string or async generator
        answer = await async_multi_topic_rag(req.question)
        # If your async_multi_topic_rag yields chunks (streaming), collect into one string:
        if hasattr(answer, "__aiter__"):
            out = ""
            async for chunk in answer:
                out += chunk
            answer = out
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

# Optionally, add a /health endpoint for enterprise monitoring
@app.get("/health")
async def health():
    return {"status": "ok"}



