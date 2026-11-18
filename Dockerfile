# The chatbot API for Hugging Face Spaces (free CPU server).
# Uploaded and started by deploy.sh; Hugging Face builds and runs this file.
FROM python:3.13-slim

# Hugging Face runs the app as user 1000, which needs its own writable folders
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /app

# CPU-only PyTorch: the free server has no GPU, and the normal version is ~2 GB bigger
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The code, owned by the app user so it can save the vector store and web cache
COPY --chown=user . .
RUN chown user /app
USER user

# Download both search models during the build, so a restart doesn't download ~4 GB first
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-m3'); CrossEncoder('BAAI/bge-reranker-v2-m3')"

# Hugging Face Spaces send visitors to port 7860
EXPOSE 7860
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "7860"]
