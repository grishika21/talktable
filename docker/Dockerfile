# TalkTable app image: the Gradio chat UI + remote MCP server (app.py).
# Built by docker-compose.yml; also used there to seed the databases.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app

# Dependencies first, so code changes don't reinstall them. CPU-only PyTorch comes from
# the extra index in requirements.txt.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image: containers start fast and work offline.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1

COPY . .

# Run as an unprivileged user. It may write only its own working files: the answer
# cache in /data and Gradio's scratch folder in /app/.gradio.
RUN useradd --create-home --uid 1000 talktable \
    && mkdir -p /data /app/.gradio \
    && chown talktable /data /app/.gradio
USER talktable

EXPOSE 7860
HEALTHCHECK --interval=10s --timeout=5s --start-period=90s --retries=6 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=4)"

CMD ["python", "app.py"]
