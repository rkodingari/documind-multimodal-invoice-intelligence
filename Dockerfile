FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ARG INSTALL_EXTRAS=model
COPY pyproject.toml README.md ./
COPY documind ./documind
RUN pip install --upgrade pip && pip install ".[${INSTALL_EXTRAS}]"
COPY . .

RUN mkdir -p /app/data/uploads && chown -R 10001:10001 /app
USER 10001

EXPOSE 8000
CMD ["uvicorn", "documind.main:app", "--host", "0.0.0.0", "--port", "8000"]
