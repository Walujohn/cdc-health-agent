FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV OPENAI_API_KEY=changeme
ENV LANGCHAIN_API_KEY=changeme
ENV LANGCHAIN_TRACING_V2=true
ENV LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
ENV LANGCHAIN_PROJECT=cdc-agent

CMD ["python", "-m", "cdc_agent.agent"]

# Then you can build and run your project anywhere with:
# docker build -t cdc-agent .
# docker run -e OPENAI_API_KEY=sk-... -e LANGCHAIN_API_KEY=lsm_... cdc-agent
