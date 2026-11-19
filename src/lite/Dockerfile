# LITE chatbot API for a 512 MB server (Render free plan).
# On Render: Language = Docker, Dockerfile Path = ./src/lite/Dockerfile, Docker Build Context = . (project folder)
# Build locally from the PROJECT folder:  docker build -f src/lite/Dockerfile -t chatbot-lite .
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    FASTEMBED_CACHE_PATH=/app/.fastembed
WORKDIR /app

COPY src/lite/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Mark the saved website copy as old, so every start reads the live website first
# (the saved copy is only used if the website cannot be downloaded)
RUN find data/web_cache -type f -exec touch -t 200001010000 {} +

# Download the two small 8-bit models during the build, so starting up needs no download
RUN python -c "import sys; sys.path[:0] = ['src/lite', 'src/shared']; import search as s; \
s.LiteEmbedder(); s.TextCrossEncoder(s.RERANK_MODEL)"

# Render tells the app which port to use in $PORT (10000 by default)
CMD ["sh", "-c", "uvicorn api:app --app-dir src/lite --host 0.0.0.0 --port ${PORT:-10000}"]
