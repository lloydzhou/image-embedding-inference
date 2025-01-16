# Image Embedding Inference (IEI)

This project provides a REST API to get embeddings from images using a pre-trained model. Its  data contract in inspired by [text-embedding-inference](https://github.com/huggingface/text-embedding-inference).

## Supported Models

Image Embedding Inference supports multiple models from Huggingface. To select a model, set the environment variable `IMAGE_EMBEDDING_MODEL` to one of the following:

- `google/vit-base-patch16-224`
- `facebook/deit-base-patch16-224`
- `microsoft/beit-base-patch16-224`
- `facebook/convnext-tiny-224`
- `facebook/convnext-base-224`
- `microsoft/swin-tiny-patch4-window7-224`
- `microsoft/swin-base-patch4-window7-224`
- `nateraw/vit-base-beans`

If no model is specified, `google/vit-base-patch16-224` will be used.

## Setup

### Docker

Build the Docker image using the following command: `docker build -t image-embedding-inference .`

To run Image Embedding Inference in a Docker container, you can use the following command: `docker run -p 8000:8000 -e IMAGE_EMBEDDING_MODEL=google/vit-base-patch16-224 image-embedding-inference`

### Python

Install the dependencies using `uv` with the following command: `uv sync`.

Activate the environment using `source .venv/bin/activate`

## Run Image Embedding Inference

To run Image Embedding Inference, you can use the following command: `IMAGE_EMBEDDING_MODEL=google/vit-base-patch16-224 gunicorn server:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --preload --bind 0.0.0.0:8000`

You can run an example request using `python examples/request_example.py`

## Example

**Request:**

The endpoint accepts a list of URLs to images.

```bash
curl -X POST 'http://localhost:8000/embed' \
--header 'Content-Type: application/json' \
-d '{
    "inputs": ["iVBORw0KGgoAAAANSUhEUgAAAogAAAQwCAYAAABmAK+YAAAMSWlDQ1BJQ0MgUHJvZm..."
    ...
    ]
}'
```

**Response:**

A list of embeddings for each image.

```json
[
    [0.8418501019477844, 0.09062539786100388, 0.21319620311260223, 0.04376870021224022, 0.5739715695381165, -0.7696743607521057, ...],
    ...
]
```