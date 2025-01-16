FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Layers can be optimized...

COPY . ./

RUN uv pip install --system --no-cache-dir .

ENV IMAGE_EMBEDDING_MODEL=google/vit-base-patch16-224
ENV PYTHONUNBUFFERED=1
ENV IEI_WORKERS=4
ENV IEI_BIND=0.0.0.0:8000

EXPOSE 8000

CMD gunicorn server:app --workers $IEI_WORKERS --worker-class uvicorn.workers.UvicornWorker --preload --bind $IEI_BIND