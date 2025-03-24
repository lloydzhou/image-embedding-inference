FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Layers can be optimized...

COPY ./pyproject.toml ./

RUN uv pip install --system --no-cache-dir .

ENV IMAGE_EMBEDDING_MODEL=nateraw/vit-base-beans
ENV PYTHONUNBUFFERED=1
ENV IEI_WORKERS=1
ENV IEI_BIND=0.0.0.0:8000

EXPOSE 8000

RUN curl -L https://huggingface.co/spaces/lloydzhou/nateraw-vit-base-beans-onnx/resolve/main/model.onnx -o /app/model.onnx
COPY ./server.py ./

CMD gunicorn server:app --workers $IEI_WORKERS --worker-class uvicorn.workers.UvicornWorker --preload --bind $IEI_BIND