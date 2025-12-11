"""基于PyTorch的YOLO检测器实现"""

import os
import threading
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

from core.config import settings
from core.logging_config import get_logger
from core.interfaces import BaseDetector, DetectionResult

logger = get_logger(__name__)


class TorchDetector(BaseDetector):
    """基于PyTorch的YOLO检测器"""

    def __init__(self, model_path: str = None, suppress_device_log: bool = False):
        if model_path is None:
            model_path = settings.MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO模型文件不存在: {model_path}")

        self.device_str = self._get_optimal_device(suppress_log=suppress_device_log)
        self.device_obj = self._get_device_object()
        self.device_info = self._get_device_info()
        self.model = None
        self.is_half = False
        self._class_names: Dict[int, str] = {}
        self._num_classes = 0
        self._inference_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"detector_{id(self)}")
        
        self._load_model(model_path, suppress_log=suppress_device_log)
        self._load_class_names(model_path, suppress_log=suppress_device_log)

    def _load_model(self, model_path: str, suppress_log: bool = False):
        """加载PyTorch模型"""
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    self.model = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    if not suppress_log:
                        logger.warning("检测到 state_dict 格式，可能需要模型架构定义")
                    self.model = checkpoint
                else:
                    self.model = checkpoint
            else:
                self.model = checkpoint

            if isinstance(self.model, torch.nn.Module):
                self.model = self.model.to(self.device_obj)
                self.model.eval()
                self.is_half = False
                try:
                    for param in self.model.parameters():
                        if param.dtype == torch.float16:
                            self.is_half = True
                            break
                except:
                    pass

                if not suppress_log:
                    if self.is_half:
                        logger.debug("模型使用半精度（float16），输入将自动转换为 half")
                    else:
                        logger.debug("模型使用全精度（float32）")
            else:
                if not suppress_log:
                    logger.warning(f"模型格式可能不是标准的 torch.nn.Module，类型: {type(self.model)}")
                if isinstance(self.model, dict) and "model" in self.model:
                    self.model = self.model["model"]
                    if isinstance(self.model, torch.nn.Module):
                        self.model = self.model.to(self.device_obj)
                        self.model.eval()
                        if not suppress_log:
                            logger.debug(f"从 checkpoint 中提取模型并移动到设备: {self.device_str}")
                    else:
                        raise ValueError(f"无法从 checkpoint 中提取有效的模型，类型: {type(self.model)}")
                else:
                    raise ValueError(f"不支持的模型格式: {type(self.model)}")

        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise

    def _load_class_names(self, model_path: str, suppress_log: bool = False):
        """加载类别名称映射"""
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict):
                if "names" in checkpoint:
                    names = checkpoint["names"]
                    if isinstance(names, dict):
                        self._class_names = {int(k): str(v) for k, v in names.items()}
                        self._num_classes = len(self._class_names)
                        if not suppress_log:
                            logger.debug(f"从模型checkpoint中读取到 {self._num_classes} 个类别")
                        return
                    elif isinstance(names, list):
                        self._class_names = {i: str(name) for i, name in enumerate(names)}
                        self._num_classes = len(self._class_names)
                        if not suppress_log:
                            logger.debug(f"从模型checkpoint中读取到 {self._num_classes} 个类别（列表格式）")
                        return
        except Exception as e:
            if not suppress_log:
                logger.debug(f"从checkpoint读取类别名称失败: {str(e)}")

        try:
            if isinstance(self.model, torch.nn.Module):
                if hasattr(self.model, "names"):
                    names = self.model.names
                    if isinstance(names, dict):
                        self._class_names = {int(k): str(v) for k, v in names.items()}
                        self._num_classes = len(self._class_names)
                        if not suppress_log:
                            logger.debug(f"从模型对象中读取到 {self._num_classes} 个类别")
                        return
                    elif isinstance(names, list):
                        self._class_names = {i: str(name) for i, name in enumerate(names)}
                        self._num_classes = len(self._class_names)
                        if not suppress_log:
                            logger.debug(f"从模型对象中读取到 {self._num_classes} 个类别（列表格式）")
                        return
        except Exception as e:
            if not suppress_log:
                logger.debug(f"从模型对象读取类别名称失败: {str(e)}")

        if settings.CLASSES_FILE:
            classes_file = Path(settings.CLASSES_FILE)
            if not classes_file.is_absolute():
                base_dir = Path(__file__).parent.parent.parent
                classes_file = base_dir / classes_file
        else:
            classes_file = Path(model_path).parent / "classes.txt"
            if not classes_file.exists():
                classes_file = Path(__file__).parent.parent.parent / "classes.txt"
        
        if classes_file.exists():
            try:
                with open(classes_file, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                    self._class_names = {i: line for i, line in enumerate(lines)}
                    self._num_classes = len(self._class_names)
                    if not suppress_log:
                        logger.debug(f"从 {classes_file} 读取到 {self._num_classes} 个类别")
                    return
            except Exception as e:
                if not suppress_log:
                    logger.warning(f"读取 classes.txt 失败: {str(e)}")

        if not suppress_log:
            logger.warning("未找到类别映射，使用默认的18类映射（向后兼容）")
        self._class_names = {
            0: "activated_smoke_detector",
            1: "car",
            2: "electric_bike",
            3: "end_water_test_device",
            4: "external_fire_hydrant",
            5: "fire_channel_marking",
            6: "fire_channel_sign",
            7: "fire_door",
            8: "fire_elevator_button",
            9: "fire_extinguisher",
            10: "fire_pump_coupler",
            11: "indoor_fire_hydrant_box",
            12: "indoor_fire_hydrant_box_closed",
            13: "mechanical_pressure_smoke_exhaust",
            14: "sprinkler_facilities",
            15: "stairwell",
            16: "sundries",
            17: "inactivated_smoke_detector",
        }
        self._num_classes = len(self._class_names)

    @property
    def class_names(self) -> Dict[int, str]:
        """获取类别名称映射"""
        return self._class_names

    def _get_optimal_device(self, suppress_log: bool = False) -> str:
        """获取最优的计算设备"""
        try:
            import torch

            try:
                import torch_musa

                if hasattr(torch, "musa") and torch.musa.is_available():
                    device_count = torch.musa.device_count()
                    if device_count > 0:
                        device_name = (
                            torch.musa.get_device_name(0) if hasattr(torch.musa, "get_device_name") else "MUSA GPU"
                        )
                        if not suppress_log:
                            logger.info(f"✓ 检测到MUSA GPU: {device_name}，将使用MUSA进行推理")
                        return "musa:0"
            except ImportError:
                if not suppress_log:
                    logger.debug("torch_musa未安装，跳过MUSA检测")
            except Exception as e:
                if not suppress_log:
                    logger.debug(f"MUSA设备检测失败: {str(e)}")

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                if not suppress_log:
                    logger.info(f"✓ 检测到GPU: {device_name} (CUDA {cuda_version})，将使用CUDA")
                return "cuda:0"

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                if not suppress_log:
                    logger.info("✓ 检测到Apple MPS，将使用mps")
                return "mps"

            if not suppress_log:
                logger.warning("⚠ 未检测到MUSA、CUDA或MPS，将使用CPU（性能较慢）")
            return "cpu"

        except ImportError as e:
            if not suppress_log:
                logger.warning(f"PyTorch未安装: {str(e)}，使用CPU")
            return "cpu"
        except Exception as e:
            if not suppress_log:
                logger.warning(f"设备检测失败: {str(e)}，使用CPU")
            return "cpu"

    def _get_device_object(self):
        """获取 torch.device 对象"""
        try:
            import torch

            if self.device_str.startswith("musa"):
                return torch.device("musa:0")
            elif self.device_str.startswith("cuda"):
                return torch.device(self.device_str)
            elif self.device_str == "mps":
                return torch.device("mps")
            else:
                return torch.device("cpu")
        except Exception as e:
            logger.warning(f"创建设备对象失败: {str(e)}，使用CPU")
            import torch
            return torch.device("cpu")

    def _get_device_info(self) -> Dict[str, Any]:
        """获取详细的硬件设备信息"""
        device_info = {
            "device": self.device_str,
            "device_type": "unknown",
            "device_name": "unknown",
            "device_count": 0,
            "driver_version": None,
            "compute_capability": None,
            "memory_total": None,
            "memory_allocated": None,
            "memory_free": None,
        }

        try:
            import torch

            if self.device_str.startswith("musa"):
                device_info["device_type"] = "MUSA GPU"
                try:
                    import torch_musa

                    if hasattr(torch, "musa") and torch.musa.is_available():
                        device_info["device_count"] = torch.musa.device_count()
                        if hasattr(torch.musa, "get_device_name"):
                            device_info["device_name"] = torch.musa.get_device_name(0)
                        else:
                            device_info["device_name"] = "MUSA GPU"
                        if hasattr(torch.musa, "get_device_properties"):
                            props = torch.musa.get_device_properties(0)
                            if hasattr(props, "total_memory"):
                                device_info["memory_total"] = f"{props.total_memory / 1024**3:.2f} GB"
                except Exception as e:
                    logger.debug(f"获取MUSA设备信息失败: {str(e)}")

            elif self.device_str.startswith("cuda"):
                device_info["device_type"] = "CUDA GPU"
                if torch.cuda.is_available():
                    device_info["device_count"] = torch.cuda.device_count()
                    device_info["device_name"] = torch.cuda.get_device_name(0)
                    device_info["driver_version"] = torch.version.cuda
                    props = torch.cuda.get_device_properties(0)
                    device_info["compute_capability"] = f"{props.major}.{props.minor}"
                    device_info["memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
                    device_info["memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
                    device_info["memory_free"] = (
                        f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB"
                    )

            elif self.device_str == "mps":
                device_info["device_type"] = "Apple MPS"
                device_info["device_name"] = "Apple Silicon GPU"
                device_info["device_count"] = 1

            else:
                device_info["device_type"] = "CPU"
                import platform
                device_info["device_name"] = platform.processor() or "CPU"
                device_info["device_count"] = 1

        except Exception as e:
            logger.warning(f"获取设备信息失败: {str(e)}")

        return device_info

    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息（接口方法）"""
        return self.device_info

    def _preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, Dict]:
        """预处理图像"""
        orig_h, orig_w = image.shape[:2]
        img_size = settings.IMGSZ
        scale = min(img_size / orig_w, img_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
        pad_h = (img_size - new_h) // 2
        pad_w = (img_size - new_w) // 2
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        rgb_float = rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb_float).permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device_obj)
        if self.is_half:
            tensor = tensor.half()
        preprocess_info = {
            "orig_h": orig_h,
            "orig_w": orig_w,
            "new_h": new_h,
            "new_w": new_w,
            "pad_h": pad_h,
            "pad_w": pad_w,
            "scale": scale,
        }

        return tensor, preprocess_info

    def _postprocess_output(
        self, output: torch.Tensor, preprocess_info: Dict, conf_threshold: float = None
    ) -> List[Tuple]:
        """后处理模型输出"""
        if conf_threshold is None:
            conf_threshold = settings.CONF_THRESHOLD

        orig_h = preprocess_info.get("orig_h", 640)
        orig_w = preprocess_info.get("orig_w", 640)
        pad_h = preprocess_info.get("pad_h", 0)
        pad_w = preprocess_info.get("pad_w", 0)
        scale = preprocess_info.get("scale", 1.0)

        if isinstance(output, (list, tuple)):
            output = output[0]

        output = output.cpu()
        if len(output.shape) == 3:
            batch_size, dim1, dim2 = output.shape
            if dim2 > dim1:
                output = output.permute(0, 2, 1)
                batch_size, num_boxes, features = output.shape
            else:
                num_boxes, features = dim1, dim2
            output = output[0]
        elif len(output.shape) == 2:
            dim1, dim2 = output.shape
            if dim2 > dim1:
                output = output.permute(1, 0)
                num_boxes, features = output.shape
            else:
                num_boxes, features = dim1, dim2
        else:
            logger.warning(f"意外的输出形状: {output.shape}")
            return []

        inferred_num_classes = max(1, features - 4)
        
        if self._num_classes > 0:
            expected_features = 4 + self._num_classes
            if features != expected_features:
                logger.warning(
                    f"输出特征数 ({features}) 与预期 ({expected_features}) 不匹配，"
                    f"使用推断的类别数: {inferred_num_classes}"
                )
            num_classes = inferred_num_classes
        else:
            num_classes = inferred_num_classes
            if self._num_classes == 0:
                logger.info(f"根据输出形状推断类别数量: {num_classes}")
                if not self._class_names:
                    self._class_names = {i: f"class_{i}" for i in range(num_classes)}
                    self._num_classes = num_classes

        boxes = output[:, :4]
        scores = output[:, 4:] if features > 5 else output[:, 4:5]
        scores_min = scores.min().item()
        scores_max = scores.max().item()

        if scores_min >= 0 and scores_max <= 1:
            scores_prob = scores
        else:
            scores_prob = torch.sigmoid(scores)

        if scores_prob.shape[1] > 1:
            class_conf, class_id = torch.max(scores_prob, dim=1)
            conf = class_conf
        else:
            conf = scores_prob.squeeze()
            class_id = torch.zeros(num_boxes, dtype=torch.long)

        mask = conf >= conf_threshold
        boxes = boxes[mask]
        conf = conf[mask]
        class_id = class_id[mask]

        if len(boxes) == 0:
            return []

        if boxes.max() <= 1.0:
            boxes = boxes * settings.IMGSZ

        center_x = boxes[:, 0]
        center_y = boxes[:, 1]
        width = boxes[:, 2]
        height = boxes[:, 3]

        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2

        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        x1 = torch.clamp(x1, 0, orig_w)
        y1 = torch.clamp(y1, 0, orig_h)
        x2 = torch.clamp(x2, 0, orig_w)
        y2 = torch.clamp(y2, 0, orig_h)

        boxes_tensor = torch.stack([x1, y1, x2, y2], dim=1)
        keep = self._nms(boxes_tensor, conf, iou_threshold=0.45)
        results = []
        for idx in keep:
            results.append(
                (float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx]), float(conf[idx]), int(class_id[idx]))
            )

        return results

    def _nms(self, boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.45) -> torch.Tensor:
        """非极大值抑制"""
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.long)

        _, indices = scores.sort(descending=True)
        keep = []

        while len(indices) > 0:
            current = indices[0]
            keep.append(current.item())

            if len(indices) == 1:
                break

            current_box = boxes[current]
            other_boxes = boxes[indices[1:]]

            x1 = torch.max(current_box[0], other_boxes[:, 0])
            y1 = torch.max(current_box[1], other_boxes[:, 1])
            x2 = torch.min(current_box[2], other_boxes[:, 2])
            y2 = torch.min(current_box[3], other_boxes[:, 3])

            inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

            area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            area_other = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
            union = area_current + area_other - inter

            iou = inter / union

            mask = iou < iou_threshold
            indices = indices[1:][mask]

        return torch.tensor(keep, dtype=torch.long)

    def _run_inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """同步推理方法"""
        with self._inference_lock:
            with torch.no_grad():
                return self.model(input_tensor)

    def detect(self, image: np.ndarray, conf_threshold: Optional[float] = None) -> List[DetectionResult]:
        """检测图像中的目标对象"""
        if conf_threshold is None:
            conf_threshold = settings.CONF_THRESHOLD

        logger.debug(f"开始检测，使用设备: {self.device_str}")
        input_tensor, preprocess_info = self._preprocess_image(image)
        output = self._run_inference(input_tensor)
        detections_list = self._postprocess_output(output, preprocess_info, conf_threshold)

        results = []
        for x1, y1, x2, y2, conf, class_id in detections_list:
            class_id = int(class_id)
            if class_id in self._class_names:
                class_name = self._class_names[class_id]
            else:
                class_name = f"class_{class_id}"
                logger.warning(f"检测到未知的 class_id: {class_id}，使用默认名称: {class_name}")

            results.append(
                DetectionResult(
                    class_name=class_name,
                    class_id=class_id,
                    confidence=float(conf),
                    bbox=[float(x1), float(y1), float(x2), float(y2)],
                )
            )

        if len(results) > settings.MAX_DET:
            results.sort(key=lambda x: x.confidence, reverse=True)
            results = results[: settings.MAX_DET]

        logger.debug(f"检测完成: 检测到 {len(results)} 个对象")
        return results
    
    async def detect_async(self, image: np.ndarray, conf_threshold: Optional[float] = None) -> List[DetectionResult]:
        """异步检测图像中的目标对象"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.detect, image, conf_threshold)

