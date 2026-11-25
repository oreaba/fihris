# Fihris images, built from this repository. Two targets:
#   online   fihris:<version>          fetches the models on first start, into /data/hf
#   offline  fihris:<version>-offline  carries the models and never uses the network
#
#   docker build -f docker/Dockerfile --target online  -t fihris:0.6 .
#   docker build -f docker/Dockerfile --target offline -t fihris:0.6-offline .
#
# Both need PostgreSQL with pgvector (FIHRIS_DATABASE_URL) and the embedding and reranker
# servers (FIHRIS_EMBED_URL, FIHRIS_RERANK_URL). Neither needs a GPU or torch.

FROM python:3.12-slim AS online
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-ara tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/fihris
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[ocr]" && rm -rf /opt/fihris
RUN useradd --create-home --uid 1000 fihris && mkdir -p /data /models \
 && chown fihris:fihris /data /models
COPY docker/entrypoint.sh /usr/local/bin/fihris-entrypoint
RUN sed -i 's/\r$//' /usr/local/bin/fihris-entrypoint && chmod 755 /usr/local/bin/fihris-entrypoint
ENV HOME=/data HF_HOME=/data/hf FIHRIS_WORK_DIR=/data/work FIHRIS_FONTS_DIR=/data/fonts \
    FIHRIS_CORPUS_ROOT=/data/documents FIHRIS_TESSERACT_CMD=/usr/bin/tesseract \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
USER fihris
WORKDIR /data
ENTRYPOINT ["fihris-entrypoint"]
CMD ["status"]

FROM online AS offline
ARG TRANSLATORS=opus-mt
ENV HF_HOME=/models/hf FIHRIS_TRANSLATOR_MODELS=${TRANSLATORS}
RUN fihris download-models --translators "${TRANSLATORS}" > /models/downloaded.json
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
