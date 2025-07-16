import os
import time
import asyncio
import mlflow
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from cdc_agent.config import CONFIG
from cdc_agent.rag import async_multi_topic_rag

API_KEY = os.getenv("API_KEY", "changeme")  # Set this in your shell/env!

def check_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

app = FastAPI(
    title="CDC Agent API",
    description="Enterprise-ready async RAG (multi-topic, CDC.gov) for health Q&A",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change for production!
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    chunk_size: int | None = None  # New: Optional chunk_size

@app.post("/ask")
async def ask_rag(
    req: QueryRequest,
    api_key: str = Depends(check_api_key)
):
    # Dynamically override chunk_size for experiment, if provided
    if req.chunk_size:
        CONFIG["chunk_size"] = req.chunk_size
    else:
        req.chunk_size = CONFIG["chunk_size"]

    start = time.time()
    with mlflow.start_run(run_name=f"api-chunksize-{req.chunk_size}"):
        mlflow.log_param("question", req.question)
        mlflow.log_param("chunk_size", req.chunk_size)
        try:
            answer = await async_multi_topic_rag(req.question)
            latency = time.time() - start
            mlflow.log_metric("latency_sec", latency)
            mlflow.log_text(str(answer), "answer.txt")
            return {
                "answer": answer,
                "latency_sec": latency,
                "chunk_size": req.chunk_size
            }
        except Exception as e:
            mlflow.log_param("error", str(e))
            return {"error": str(e)}

@app.post("/ask/stream")
async def ask_rag_stream(
    req: QueryRequest,
    api_key: str = Depends(check_api_key)
):
    if req.chunk_size:
        CONFIG["chunk_size"] = req.chunk_size
    else:
        req.chunk_size = CONFIG["chunk_size"]

    async def event_generator():
        with mlflow.start_run(run_name=f"api-stream-chunksize-{req.chunk_size}"):
            mlflow.log_param("question", req.question)
            mlflow.log_param("chunk_size", req.chunk_size)
            try:
                rag_chunks = async_multi_topic_rag(req.question)
                answer_accum = ""
                async for chunk in rag_chunks:
                    answer_accum += chunk
                    yield f"data: {chunk}\n\n"
                mlflow.log_text(answer_accum, "streamed_answer.txt")
                yield "data: [DONE]\n\n"
            except Exception as e:
                mlflow.log_param("error", str(e))
                yield f"data: [ERROR] {str(e)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status": "ok"}



