import asyncio
import os
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, APIRouter, Depends
import base64
from PIL import Image
import io
import numpy as np
from functools import lru_cache
from datetime import datetime
import logging
import onnxruntime as ort
import json

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

    def __init__(self, model_path: str, config_path: str = None):
        # 设置ONNX运行时选项
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # 检查CUDA可用性并设置提供者
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        try:
            # 加载ONNX模型
            self.model = ort.InferenceSession(model_path, options, providers=providers)
            
            # 加载模型配置（图像尺寸和隐藏层大小）
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                # 默认配置
                self.config = {
                    "image_size": {"height": 224, "width": 224},
                    "hidden_size": 768  # ViT-base的典型隐藏层大小
                }
                
            # 获取模型输入名称
            self.input_name = self.model.get_inputs()[0].name
            self.output_name = self.model.get_outputs()[0].name
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_path}: {str(e)}")

    @classmethod
    def is_valid_model_name(cls, model_name: str) -> bool:
        return model_name in cls.MODELS or os.path.exists(model_name)

    @classmethod
    def from_model_name(cls, model_name: str) -> "ModelManager":
        if not cls.is_valid_model_name(model_name):
            raise ValueError(f"Invalid model name: {model_name}")
        
        # 如果指定的是模型ID而不是路径，使用默认ONNX路径
        if model_name in cls.MODELS:
            model_path = os.getenv("ONNX_MODEL_PATH", "onnx/model.onnx")
            config_path = os.getenv("MODEL_CONFIG_PATH", "onnx/config.json")
        else:
            model_path = model_name
            config_dir = os.path.dirname(model_name)
            config_path = os.path.join(config_dir, "model_config.json")
            
        return cls(model_path, config_path)


class EmbeddingRequest(BaseModel):
    inputs: list[str]  # base64 encoded images


async def decode_base64_image(base64_string: str) -> Image.Image:
    if "how big" == base64_string:
        imarray = np.random.rand(224,224,3) * 255
        return Image.fromarray(imarray.astype('uint8')).convert('RGBA')
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


def preprocess_image(image: Image.Image, target_size: tuple) -> np.ndarray:
    """处理图像适应ONNX模型输入"""
    # 调整图像大小
    image = image.resize(target_size)
    
    # 转换为RGB（如果有Alpha通道）
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # 转换为NumPy数组并归一化
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # 标准化 (使用ImageNet标准均值和方差) - 确保使用float32类型
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    image_array = (image_array - mean) / std
    
    # 转换为NCHW格式 (batch, channels, height, width)
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)
    
    # 确保输出是float32类型
    image_array = image_array.astype(np.float32)
    
    return image_array


@lru_cache()
def get_model_manager() -> ModelManager:
    model_name = os.getenv("IMAGE_EMBEDDING_MODEL", "google/vit-base-patch16-224")
    model_path = os.getenv("ONNX_MODEL_PATH", "model.onnx")
    
    # 如果提供的是模型ID而不是实际路径，使用默认ONNX路径
    if not os.path.exists(model_name):
        model_name = model_path
        
    LOGGER.info(f"Loading model: {model_name}")
    LOGGER.info(f"ONNX providers: {ort.get_available_providers()}")

    mm = ModelManager.from_model_name(model_name)
    
    LOGGER.info(f"Embeddings with size: {mm.config['hidden_size'] if 'hidden_size' in mm.config else 'Unknown'}")

    return mm


@router.get("/health")
async def health():
    return {"message": "OK"}


async def embed_image(image_base64: str, model_manager: ModelManager) -> list[float]:
    LOGGER.info(f"Embedding image")

    image = await decode_base64_image(image_base64)
    loop = asyncio.get_event_loop()

    # 获取目标图像尺寸
    target_size = (
        model_manager.config["image_size"]["width"],
        model_manager.config["image_size"]["height"]
    )
    
    # 预处理图像
    image_tensor = await loop.run_in_executor(
        None, lambda: preprocess_image(image, target_size)
    )
    
    # 执行ONNX模型推理
    ort_inputs = {model_manager.input_name: image_tensor}
    
    outputs = await loop.run_in_executor(
        None, lambda: model_manager.model.run([model_manager.output_name], ort_inputs)
    )
    
    # 处理输出以获取嵌入向量（假设输出是最后隐藏状态的平均值）
    # 注意：这里的处理可能需要根据具体ONNX模型的输出格式进行调整
    embeddings = outputs[0].mean(axis=1).flatten().tolist()

    LOGGER.info("Embeddings generated.")
    return embeddings


async def bounded_embed_image(
    semaphore: asyncio.Semaphore, image_base64: str, model_manager: ModelManager
) -> list[float]:
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
            tasks.append(
                asyncio.ensure_future(
                    bounded_embed_image(semaphore, image_input, model_manager)
                )
            )
        embeddings = await asyncio.gather(*tasks)

        LOGGER.info(f"Tasks completed ({len(tasks)})")

        total_time = (datetime.now() - now).total_seconds()
        LOGGER.info(
            f"Embeddings generated in {total_time} seconds. | Avg time per image: {total_time / len(request.inputs)} seconds."
        )
        return embeddings

    except ValueError as e:
        # LOGGER.error("error %r", request.inputs)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Embedding generation failed: {str(e)}"
        )


def create_app():
    get_concurrency_level()
    # Current setup doesn't support shared model manager.

    app = FastAPI(
        title="Image Embedding Inference API",
        description="API for generating image embeddings.",
        version="0.0.1",
    )
    app.include_router(router)

    return app


app = create_app()
