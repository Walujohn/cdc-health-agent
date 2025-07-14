import mlflow
from cdc_agent.config import CONFIG

def start_agent_run(run_name, extra_tags=None):
    tags = {"project": "cdc-agent", **(extra_tags or {})}
    mlflow.start_run(run_name=run_name)
    mlflow.set_tags(tags)
    mlflow.log_params(CONFIG)

def log_agent_metrics(question, topic, summary, cdc_info):
    mlflow.log_param("question", question)
    mlflow.log_param("topic", topic)
    mlflow.log_metric("summary_length", len(summary or ""))
    mlflow.log_text(summary, "summary.txt")
    mlflow.log_text(cdc_info, "cdc_info.txt")

def log_agent_error(error_message):
    mlflow.log_text(str(error_message), "error.txt")
