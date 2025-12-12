"""YOLO 检测服务 API 层"""

import io
import threading
import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Tuple
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


class ModelDetectionResult(BaseModel):
    """单个模型的检测结果"""
    model_name: str
    detections: Dict[str, int]
    total_objects: int
    details: List[DetectionDetail]
    error: Optional[str] = None


class MultiModelDetectionResponse(BaseModel):
    """多模型检测响应"""
    results: List[ModelDetectionResult]


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


def _process_detection_results(detection_results: List) -> Tuple[Dict[str, int], List[DetectionDetail]]:
    """处理检测结果，返回统计信息和详细信息"""
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
    return dict(detections), details


async def _detect_single_model(
    detector: BaseDetector,
    model_name: str,
    frame_bgr: np.ndarray,
    conf_threshold: float,
) -> ModelDetectionResult:
    """单个模型的检测任务"""
    try:
        detection_results = await detector.detect_async(frame_bgr, conf_threshold=conf_threshold)
        detections, details = _process_detection_results(detection_results)
        
        return ModelDetectionResult(
            model_name=model_name,
            detections=detections,
            total_objects=sum(detections.values()),
            details=details,
            error=None,
        )
    except Exception as e:
        logger.error(f"模型 '{model_name}' 检测失败: {str(e)}")
        return ModelDetectionResult(
            model_name=model_name,
            detections={},
            total_objects=0,
            details=[],
            error=str(e),
        )


@router.post("/detect", response_model=MultiModelDetectionResponse)
async def detect_objects(
    file: UploadFile = File(..., description="图片文件"),
    model_names: str = Form(..., description="模型名称数组（JSON格式，支持单个或多个模型，如：[\"model1\"] 或 [\"model1\", \"model2\"]）"),
    conf_thresholds: Optional[str] = Form(None, description="每个模型的阈值映射（JSON格式，可选，如：{\"model1\": 0.5, \"model2\": 0.6}，未指定的模型使用默认值0.5）"),
):
    """
    检测图片中的目标对象
    
    统一使用多模型模式，支持传入单个或多个模型：
    - model_names: JSON数组格式，如 ["model1"] 或 ["model1", "model2"]
    - conf_thresholds: JSON对象格式，可选，为每个模型指定阈值，如 {"model1": 0.5, "model2": 0.6}
    - 返回 MultiModelDetectionResponse，包含每个模型的检测结果
    """
    try:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="文件必须是图片格式")

        # 解析模型名称列表
        try:
            model_names_list = json.loads(model_names)
            if not isinstance(model_names_list, list):
                raise ValueError("model_names 必须是JSON数组格式")
            if len(model_names_list) == 0:
                raise ValueError("model_names 数组不能为空")
            # 验证数组元素都是字符串
            if not all(isinstance(name, str) for name in model_names_list):
                raise ValueError("model_names 数组中的元素必须是字符串")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="model_names 必须是有效的JSON数组格式")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # 解析阈值配置
        thresholds_dict: Dict[str, float] = {}
        default_threshold = 0.5
        if conf_thresholds:
            try:
                thresholds_dict = json.loads(conf_thresholds)
                if not isinstance(thresholds_dict, dict):
                    raise ValueError("conf_thresholds 必须是JSON对象格式")
                # 验证阈值都是数字
                for key, value in thresholds_dict.items():
                    if not isinstance(value, (int, float)):
                        raise ValueError(f"conf_thresholds 中的阈值必须是数字，但 '{key}' 的值为 {type(value).__name__}")
                    if not (0 <= value <= 1):
                        raise ValueError(f"conf_thresholds 中的阈值必须在0-1之间，但 '{key}' 的值为 {value}")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="conf_thresholds 必须是有效的JSON对象格式")
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        # 读取和预处理图片
        image_data = await file.read()
        pil_image = Image.open(io.BytesIO(image_data))
        frame_rgb = np.array(pil_image)
        
        if len(frame_rgb.shape) == 2:
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2RGB)
        elif frame_rgb.shape[2] == 4:
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGBA2RGB)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 验证模型存在性
        model_manager = get_model_manager()
        available_models = model_manager.list_models()
        invalid_models = [name for name in model_names_list if name not in available_models]
        if invalid_models:
            raise HTTPException(
                status_code=404,
                detail=f"以下模型不存在: {invalid_models}。可用模型: {available_models}"
            )

        # 为每个模型准备检测任务
        detection_tasks = []
        for model_name_item in model_names_list:
            detector = model_manager.get_model(model_name_item)
            if not detector:
                continue  # 理论上不会到这里，因为已经验证过了
            
            # 获取该模型的阈值，优先使用conf_thresholds中的配置，否则使用默认值
            model_threshold = thresholds_dict.get(model_name_item, default_threshold)
            detection_tasks.append(
                _detect_single_model(detector, model_name_item, frame_bgr, model_threshold)
            )

        # 并行执行所有模型的检测
        results = await asyncio.gather(*detection_tasks)

        # 构建响应
        response = MultiModelDetectionResponse(results=list(results))

        # 保存图片（如果启用）
        if settings.SAVE_TMP_ENABLED:
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

                    total_objects = sum(r.total_objects for r in results)
                    model_count = len(results)
                    if total_objects > 0:
                        logger.info(f"图片已保存到临时目录: {save_path} ({model_count}个模型检测，共检测到 {total_objects} 个对象)")
                    else:
                        logger.info(f"图片已保存到临时目录: {save_path} ({model_count}个模型检测，未检测到目标)")
                except Exception as e:
                    logger.warning(f"保存图片到临时目录失败: {str(e)}")
            
            asyncio.create_task(save_image_async())

        # 记录日志
        successful_models = [r.model_name for r in results if r.error is None]
        failed_models = [r.model_name for r in results if r.error is not None]
        model_count = len(results)
        if model_count == 1:
            model_name = results[0].model_name
            threshold = thresholds_dict.get(model_name, default_threshold)
            if results[0].error is None:
                logger.info(
                    f"API响应 [模型: {model_name}, 阈值: {threshold}]: {results[0].total_objects} 个对象, "
                    f"{len(results[0].detections)} 种类型, {len(results[0].details)} 个详细信息"
                )
            else:
                logger.warning(f"API响应 [模型: {model_name}]: 检测失败 - {results[0].error}")
        else:
            logger.info(
                f"API响应: {model_count}个模型, 成功 {len(successful_models)} 个 ({', '.join(successful_models)}), "
                f"失败 {len(failed_models)} 个 ({', '.join(failed_models) if failed_models else '无'})"
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"检测接口错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"检测失败: {str(e)}")
