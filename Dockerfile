FROM python:3.11-slim

WORKDIR /app

# 预装系统依赖（PDF 处理、MiniIO 客户端等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（与 pyproject.toml 保持同步）
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    fastapi uvicorn "fastapi>=0.135.1" \
    "langchain>=1.2.12" "langchain-community>=0.4.1" \
    "langchain-openai>=1.1.11" "langchain-text-splitters>=1.0.0" \
    "langgraph>=1.1.2" \
    loguru "loguru>=0.7.3" \
    "magic-pdf>=1.3.12" \
    "minio>=7.2.20" \
    "modelscope>=1.35.0" \
    neo4j "neo4j>=5.28.2" \
    numpy pandas \
    "pymilvus[model]>=2.6.10" "pymilvus-model>=0.3.2" \
    pymongo "pymongo>=4.16.0" \
    python-dotenv "python-multipart>=0.0.22" \
    regex requests \
    torch torchvision \
    transformers "transformers>=4.57.6" \
    "flagembedding>=1.3.5" grandalf \
    "ragas>=0.4.3" \
    "openai-agents>=0.4.2" \
    starlette

COPY app/ ./app/
COPY prompts/ ./prompts/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# app-import 监听 0.0.0.0:8000，app-query 监听 0.0.0.0:8001
EXPOSE 8000 8001
