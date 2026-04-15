FROM python:3.11-slim-bookworm

WORKDIR /app

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_EXTRA_INDEX_URL=https://pypi.org/simple
ARG PIP_DEFAULT_TIMEOUT=1200
ARG PIP_RETRIES=10

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV DEBIAN_FRONTEND=noninteractive

# 预装系统依赖（PDF 处理、MiniIO 客户端等）
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=60 -o Acquire::https::Timeout=60 update \
    && apt-get install -y --fix-missing --no-install-recommends \
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
RUN python - <<'PY' > /tmp/requirements.txt
import tomllib
from pathlib import Path

data = tomllib.loads(Path("pyproject.toml").read_text())
for dependency in data["project"]["dependencies"]:
    print(dependency)
PY
RUN python -m pip install \
    --upgrade \
    --default-timeout="${PIP_DEFAULT_TIMEOUT}" \
    --retries="${PIP_RETRIES}" \
    --index-url="${PIP_INDEX_URL}" \
    --extra-index-url="${PIP_EXTRA_INDEX_URL}" \
    pip setuptools wheel \
    && python -m pip install \
    --no-cache-dir \
    --prefer-binary \
    --default-timeout="${PIP_DEFAULT_TIMEOUT}" \
    --retries="${PIP_RETRIES}" \
    --index-url="${PIP_INDEX_URL}" \
    --extra-index-url="${PIP_EXTRA_INDEX_URL}" \
    -r /tmp/requirements.txt

COPY app/ ./app/
COPY docs/ ./docs/
COPY prompts/ ./prompts/
COPY reports/ ./reports/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# app-import 监听 0.0.0.0:8000，app-query 监听 0.0.0.0:8001
EXPOSE 8000 8001
