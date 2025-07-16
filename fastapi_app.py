import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from cdc_agent.rag import async_multi_topic_rag

app = FastAPI(
    title="CDC Agent API",
    description="Enterprise-ready async RAG (multi-topic, CDC.gov) for health Q&A",
    version="0.1.0"
)

# Enable CORS for frontend/UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict as needed for production
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_rag(req: QueryRequest):
    """
    Traditional endpoint: waits for entire answer, then returns JSON.
    """
    try:
        answer = await async_multi_topic_rag(req.question)
        # If your RAG ever yields chunks (future streaming), collect into one string:
        if hasattr(answer, "__aiter__"):
            out = ""
            async for chunk in answer:
                out += chunk
            answer = out
        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask/stream")
async def ask_rag_stream(req: QueryRequest):
    """
    Streaming endpoint: yields partial output chunks (SSE).
    Useful for chat UIs and real-time feedback.
    """
    async def event_generator():
        try:
            rag_chunks = async_multi_topic_rag(req.question)
            async for chunk in rag_chunks:
                # Format as Server-Sent Events (works with JS/EventSource, Gradio, etc.)
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status": "ok"}



