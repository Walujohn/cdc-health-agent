import time
import mlflow

def cdc_guidance_drift_job():
    """
    Can be triggered as a background task from FastAPI or scheduled via cron, Airflow, etc.
    Logs results and metadata to MLflow.
    """
    today = time.strftime("%Y-%m-%d")
    mlflow.set_experiment("cdc_guidance_drift")
    with mlflow.start_run(run_name=f"drift_{today}_test"):
        # Can extend this: fetch CDC content, compare to previous snapshot, etc.
        mlflow.log_param("test_param", "hello_world")
        mlflow.set_tag("triggered_by", "api")
        mlflow.log_metric("random_metric", 123)
        mlflow.log_text("This is a test/stub run.", "output.txt")
        print("[DRIFT JOB] Ran a stub CDC drift experiment via FastAPI.")

# --- Add more experiment jobs below as functions ---
# For example:
def batch_latency_sweep():
    """
    Example: Run latency experiments across multiple chunk sizes and log results to MLflow.
    """
    from cdc_agent.config import CONFIG
    from cdc_agent.rag import async_multi_topic_rag
    import asyncio

    mlflow.set_experiment("cdc_latency_sweep")
    chunk_sizes = [200, 400, 600, 800]
    question = "What are the symptoms of covid?"

    for chunk_size in chunk_sizes:
        CONFIG["chunk_size"] = chunk_size
        with mlflow.start_run(run_name=f"latency-chunksize-{chunk_size}"):
            mlflow.log_param("chunk_size", chunk_size)
            mlflow.log_param("question", question)
            start = time.time()
            answer = asyncio.run(async_multi_topic_rag(question))
            latency = time.time() - start
            mlflow.log_metric("latency_sec", latency)
            mlflow.log_text(str(answer), "answer.txt")
            print(f"[BATCH] chunk_size={chunk_size}, latency={latency:.2f}s")

# Usage example:
# Call `cdc_guidance_drift_job()` or `batch_latency_sweep()` from endpoint or scheduler!
