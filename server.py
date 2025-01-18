import asyncio
import os
import torch
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, APIRouter, Depends
import base64
from PIL import Image
import io
from transformers import AutoImageProcessor, AutoModel  # type: ignore
import numpy as np
from functools import lru_cache
from datetime import datetime
import logging

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()],
)

router = APIRouter()

@lru_cache()
def get_concurrency_level() -> int:
    level = os.getenv("IEI_CONCURRENCY", 1000)
    LOGGER.info(f"Concurrency level set to: {level}")
    return int(level)

class ModelManager:
    MODELS = [
        "google/vit-base-patch16-224",
        "facebook/deit-base-patch16-224",
        "microsoft/beit-base-patch16-224",
        "facebook/convnext-tiny-224",
        "facebook/convnext-base-224",
        "microsoft/swin-tiny-patch4-window7-224",
        "microsoft/swin-base-patch4-window7-224",
        "nateraw/vit-base-beans",
    ]

    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = AutoModel.from_pretrained(
                model_name, add_pooling_layer=False
            ).to(self.device)
            self.processor = AutoImageProcessor.from_pretrained(
                model_name, use_fast=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {str(e)}")

    @classmethod
    def is_valid_model_name(cls, model_name: str) -> bool:
        return model_name in cls.MODELS

    @classmethod
    def from_model_name(cls, model_name: str) -> "ModelManager":
        if not cls.is_valid_model_name(model_name):
            raise ValueError(f"Invalid model name: {model_name}")
        return cls(model_name)


class EmbeddingRequest(BaseModel):
    inputs: list[str]  # base64 encoded images


async def decode_base64_image(base64_string: str) -> Image.Image:
    if not base64_string:
        raise ValueError("Empty image data")

    try:
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
        image_data = base64.b64decode(base64_string)
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(
            None, lambda: Image.open(io.BytesIO(image_data))
        )
        return image
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")


@lru_cache()
def get_model_manager() -> ModelManager:
    model_name = os.getenv("IMAGE_EMBEDDING_MODEL", "google/vit-base-patch16-224")
    LOGGER.info(f"Loading model: {model_name}")
    LOGGER.info(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    mm = ModelManager.from_model_name(model_name)
    
    LOGGER.info(f"Embeddings with size: {mm.model.config.hidden_size}")
    
    return mm


@router.get("/health")
async def health():
    return {"message": "OK"}


async def embed_image(image_base64: str, model_manager: ModelManager) -> list[float]:
    LOGGER.info(f"Embedding image")

    image = await decode_base64_image(image_base64)
    image_array = np.array(image)
    loop = asyncio.get_event_loop()

    if image_array.ndim != 3:
        raise ValueError("Image must be RGB or RGBA")

    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    # Get required input size from processor
    input_size = model_manager.processor.size
    image = Image.fromarray(image_array).resize(
        (input_size["width"], input_size["height"])
    )

    inputs = await loop.run_in_executor(
        None, lambda: model_manager.processor(images=image, return_tensors="pt")
    )
    inputs = {k: v.to(model_manager.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = await loop.run_in_executor(
            None, lambda: model_manager.model(**inputs)
        )
        embeddings = (
            outputs.last_hidden_state.mean(dim=1).cpu().detach().numpy().tolist()[0]
        )

    LOGGER.info("Embeddings generated.")
    return embeddings

async def bounded_embed_image(semaphore: asyncio.Semaphore, image_base64: str, model_manager: ModelManager) -> list[float]:
    async with semaphore:
        return await embed_image(image_base64, model_manager)

@router.post("/embed", response_model=list[list[float]])
async def embed(
    request: EmbeddingRequest, model_manager: ModelManager = Depends(get_model_manager)
):
    try:
        LOGGER.info(f"Embedding {len(request.inputs)} images...")
        now = datetime.now()

        tasks = []
        semaphore = asyncio.Semaphore(get_concurrency_level())
        for image_input in request.inputs:
            tasks.append(asyncio.ensure_future(bounded_embed_image(semaphore, image_input, model_manager)))
        embeddings = await asyncio.gather(*tasks)

        LOGGER.info(f"Tasks completed ({len(tasks)})")

        total_time = (datetime.now() - now).total_seconds()
        LOGGER.info(
            f"Embeddings generated in {total_time} seconds. | Avg time per image: {total_time / len(request.inputs)} seconds."
        )
        return embeddings

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Embedding generation failed: {str(e)}"
        )


def create_app():
    get_concurrency_level()

    app = FastAPI(
        title="Image Embedding Inference API",
        description="API for generating image embeddings.",
        version="0.0.1",
    )
    app.include_router(router)

    return app


app = create_app()
