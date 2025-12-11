"""YOLO 检测服务 API 层"""

import io
import threading
from typing import Dict, List, Optional, Any
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel

from core.config import settings
from core.logging_config import get_logger
from services.model_manager import get_model_manager
from core.interfaces import BaseDetector

logger = get_logger(__name__)

router = APIRouter()


class DetectionDetail(BaseModel):
    """检测详细信息"""
    class_name: str
    class_id: int
    confidence: float
    bbox: List[float]


class DetectionResponse(BaseModel):
    """检测响应"""

    detections: Dict[str, int]
    total_objects: int
    details: List[DetectionDetail]


@router.get("/models")
async def list_models():
    """列出所有已加载的模型"""
    try:
        model_manager = get_model_manager()
        models = model_manager.list_models()
        return {
            "status": "success",
            "models": models,
            "count": len(models),
        }
    except Exception as e:
        logger.error(f"获取模型列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")


@router.get("/device-info")
async def get_device_info(model_name: Optional[str] = None):
    """获取硬件设备信息"""
    try:
        model_manager = get_model_manager()
        
        if model_name:
            detector = model_manager.get_model(model_name)
            if not detector:
                raise HTTPException(status_code=404, detail=f"模型 '{model_name}' 不存在")
        else:
            detector = model_manager.get_default_model()
            if not detector:
                raise HTTPException(status_code=500, detail="未找到任何已加载的模型")
        
        return {
            "status": "success",
            "model_name": model_name or "default",
            "device_info": detector.get_device_info(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取设备信息失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取设备信息失败: {str(e)}")


@router.post("/detect", response_model=DetectionResponse)
async def detect_objects(
    file: UploadFile = File(..., description="图片文件"),
    model_name: str = Form(..., description="模型名称（必填）"),
    conf_threshold: float = Form(0.5, description="检测阈值（可选），默认值0.5"),
):
    """检测图片中的目标对象"""
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="文件必须是图片格式")

        image_data = await file.read()
        pil_image = Image.open(io.BytesIO(image_data))
        frame_rgb = np.array(pil_image)
        
        if len(frame_rgb.shape) == 2:
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)
        elif frame_rgb.shape[2] == 4:
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGBA2RGB)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        model_manager = get_model_manager()
        detector = model_manager.get_model(model_name)
        if not detector:
            available_models = model_manager.list_models()
            raise HTTPException(
                status_code=404,
                detail=f"模型 '{model_name}' 不存在。可用模型: {available_models}"
            )

        detection_results = await detector.detect_async(frame_bgr, conf_threshold=conf_threshold)

        detections = defaultdict(int)
        details = []
        for result in detection_results:
            detections[result.class_name] += 1
            details.append(
                DetectionDetail(
                    class_name=result.class_name,
                    class_id=result.class_id,
                    confidence=result.confidence,
                    bbox=result.bbox,
                )
            )

        response = DetectionResponse(
            detections=dict(detections),
            total_objects=sum(detections.values()),
            details=details,
        )

        if settings.SAVE_TMP_ENABLED:
            import asyncio
            
            async def save_image_async():
                try:
                    save_dir = Path(settings.SAVE_TMP_DIR)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    original_filename = file.filename or "image"
                    save_filename = f"{timestamp}_{original_filename}"
                    save_path = save_dir / save_filename

                    loop = asyncio.get_event_loop()
                    
                    def write_file():
                        save_dir.mkdir(parents=True, exist_ok=True)
                        with open(save_path, "wb") as f:
                            f.write(image_data)
                    
                    await loop.run_in_executor(None, write_file)

                    if response.total_objects > 0:
                        logger.info(f"图片已保存到临时目录: {save_path} (检测到 {response.total_objects} 个对象)")
                    else:
                        logger.info(f"图片已保存到临时目录: {save_path} (未检测到目标)")
                except Exception as e:
                    logger.warning(f"保存图片到临时目录失败: {str(e)}")
            
            asyncio.create_task(save_image_async())
        if response.details:
            confidence_info = ", ".join(
                [f"{detail.class_name}({detail.confidence:.3f})" for detail in response.details]
            )
            logger.info(
                f"API响应 [模型: {model_name}, 阈值: {conf_threshold}]: {response.total_objects} 个对象, {len(detections)} 种类型, {len(response.details)} 个详细信息, 检测结果: [{confidence_info}]"
            )
        else:
            logger.info(
                f"API响应 [模型: {model_name}, 阈值: {conf_threshold}]: {response.total_objects} 个对象, {len(detections)} 种类型, {len(response.details)} 个详细信息"
            )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"检测接口错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")
